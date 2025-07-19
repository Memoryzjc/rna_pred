import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add, scatter_mean

class EGNNLayer(MessagePassing):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGNNLayer, self).__init__(aggr='add')  # Use 'add' aggregation
        # 基本参数设置
        input_edge = input_nf * 2  # 边特征维度：源节点特征 + 目标节点特征
        self.residual = residual   # 是否使用残差连接
        self.attention = attention # 是否使用注意力机制
        self.normalize = normalize # 是否对坐标差向量进行归一化
        self.coords_agg = coords_agg  # 坐标聚合方式：'mean' 或 'sum'
        self.tanh = tanh          # 是否在坐标更新中使用tanh激活
        self.epsilon = 1e-8       # 数值稳定性常数
        edge_coords_nf = 1        # 边的坐标特征维度（距离标量）

        # 边特征处理网络：处理节点对之间的关系信息
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # 节点特征更新网络：基于聚合的边信息更新节点特征
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # 坐标更新网络：生成坐标更新的标量权重（关键：无偏置保证等变性）
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)  # 小的初始化权重确保稳定性

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())  # 可选：限制坐标更新幅度
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # 可选的注意力机制
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())  # 生成0-1之间的注意力权重

    def forward(self, h, coords, edge_index, edge_attr=None):
        """
        前向传播函数
        Args:
            h: 节点特征 [num_nodes, input_nf]
            coords: 节点坐标 [num_nodes, 3]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edges_in_d] (可选)
        Returns:
            updated_x: 更新后的节点特征
            updated_pos: 更新后的坐标
        """
        # 如果没有边属性，创建空的边属性
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), 0, device=h.device)
        
        # 计算坐标差和距离
        row, col = edge_index
        coord_diff = coords[row] - coords[col]

        if self.normalize:
            norm = torch.norm(coord_diff, dim=1, keepdim=True)
            coord_diff = coord_diff / (norm + self.epsilon)

        dist = torch.sum((coords[row] - coords[col]) ** 2, dim=-1, keepdim=True)

        # 执行消息传递
        updated_h, coord_update = self.propagate(
            edge_index, 
            x=h, 
            pos=coords, 
            edge_attr=edge_attr, 
            coord_diff=coord_diff, 
            dist=dist,
            size=None
        )
        
        # 更新坐标
        updated_coords = coords + coord_update

        return updated_h, updated_coords

    def message(self, x_i, x_j, edge_attr, coord_diff, dist):
        """
        消息函数：计算边上的消息
        """
        # 拼接节点特征、距离和边属性
        if edge_attr.size(1) > 0:
            edge_input = torch.cat([x_i, x_j, dist, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([x_i, x_j, dist], dim=-1)
        
        # 计算边特征
        edge_feat = self.edge_mlp(edge_input)
        
        # 应用注意力机制（如果启用）
        if self.attention:
            att_val = self.att_mlp(edge_feat)
            edge_feat = edge_feat * att_val
        
        # 计算坐标更新
        coord_weight = self.coord_mlp(edge_feat)
        coord_update = coord_diff * coord_weight
        
        return edge_feat, coord_update

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        聚合函数：聚合来自邻居的消息
        """
        edge_feat, coord_update = inputs
        
        # 聚合节点特征
        agg_feat = scatter_add(edge_feat, index, dim=0, dim_size=dim_size)
        
        # 聚合坐标更新
        if self.coords_agg == 'sum':
            agg_coord = scatter_add(coord_update, index, dim=0, dim_size=dim_size)
        elif self.coords_agg == 'mean':
            agg_coord = scatter_mean(coord_update, index, dim=0, dim_size=dim_size)
        else:
            raise ValueError(f"Invalid coords_agg: {self.coords_agg}")
        
        return agg_feat, agg_coord

    def update(self, aggr_out, x):
        """
        更新函数：基于聚合的消息更新节点特征
        """
        agg_feat, agg_coord = aggr_out
        
        # 更新节点特征
        node_input = torch.cat([x, agg_feat], dim=-1)
        updated_x = self.node_mlp(node_input)
        
        # 应用残差连接
        if self.residual:
            updated_x = x + updated_x
        
        return updated_x, agg_coord


class EGNN(nn.Module):
    '''
    Implementation of the EGNN (Equivariant Graph Neural Network)
    '''
    def __init__(self, in_channels, hidden_dim, out_channels, 
                 edge_dim=1, n_layers=4, 
                 act_fn=nn.SiLU(), residual=True, 
                 attention=False, normalize=False, 
                 coords_agg='mean', tanh=False):
        super(EGNN, self).__init__()
        self.embedding_in = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(
                input_nf=hidden_dim,
                output_nf=hidden_dim,
                hidden_nf=hidden_dim,
                edges_in_d=edge_dim,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                coords_agg=coords_agg,
                tanh=tanh
            ) for _ in range(n_layers)
        ])
        self.embedding_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, h, coords, edge_index, edge_attr=None):
        """
        EGNN前向传播函数
        Args:
            h: 输入节点特征 [num_nodes, in_channels] -> [batch_size*n_nodes, node_features]
            coords: 输入节点坐标 [num_nodes, 3] -> [batch_size*n_nodes, 3]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim] (可选)
        Returns:
            h: 输出节点特征 [num_nodes, out_channels] -> [batch_size*n_nodes, out_channels]
            coords: 更新后的节点坐标 [num_nodes, 3] -> [batch_size*n_nodes, 3]
        """
        h = self.embedding_in(h)

        for layer in self.layers:
            h, coords_update = layer(h, coords, edge_index, edge_attr)
            coords = coords + coords_update

        h = self.embedding_out(h)
        
        return h, coords


class Encoder(nn.Module):
    def __init__(self, seq_embedding_dim, hidden_dim, out_node_dim, edge_dim=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(Encoder, self).__init__()
        self.seq_embedding = nn.Embedding(5, seq_embedding_dim)
        self.encoder_egnn = EGNN(seq_embedding_dim, hidden_dim, out_node_dim, edge_dim=edge_dim, act_fn=act_fn, n_layers=n_layers,
                                 residual=residual, attention=attention, normalize=normalize, tanh=tanh)
        self.to(device)
    
    def forward(self, data):
        seq = data.x  # [num_nodes, 1] -> RNA序列索引
        # Handle both one-hot and index formats
        if seq.dim() == 2 and seq.size(1) > 1:
            # One-hot encoded input - convert to indices
            seq = torch.argmax(seq, dim=1)  # [num_nodes]
        else:
            # Already indices
            seq = seq.squeeze(-1) if seq.dim() == 2 else seq  # [num_nodes]
        
        h = self.seq_embedding(seq)  # [num_nodes, seq_embedding_dim]
        coords = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # dimensions:
        # h: [num_nodes, seq_embedding_dim]
        # coords: [num_nodes, 3]
        h, coords = self.encoder_egnn(h, coords, edge_index, edge_attr)
        return h, coords



class Decoder(nn.Module):
    def __init__(self, in_node_dim, hidden_dim, out_node_dim, edge_dim=1, n_layers=4,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, tanh=False):
        super(Decoder, self).__init__()
        self.egnn = EGNN(in_node_dim, hidden_dim, out_node_dim, edge_dim=edge_dim, n_layers=n_layers,
                         act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh)

    def forward(self, z, pos, edge_index, edge_attr=None):
        return self.egnn(z, pos, edge_index, edge_attr)


class VQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQEmbedding, self).__init__()
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)

    def forward(self, h):
        # 展平输入以便计算距离
        original_shape = h.shape
        h_flat = h.detach().reshape(-1, h.size(-1))
        
        # 计算到码本向量的距离
        distances = torch.cdist(h_flat, self.embedding.weight, p=2) ** 2
        indices = torch.argmin(distances.float(), dim=-1)
        quantized = self.embedding(indices).view(original_shape)

        # 计算损失
        codebook_loss = F.mse_loss(h.detach(), quantized)
        e_latent_loss = F.mse_loss(h, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        # 使用straight-through estimator
        quantized = h + (quantized - h).detach()
        return quantized, commitment_loss, codebook_loss


class VQEGNN(nn.Module):
    def __init__(self, seq_embedding_dim, hidden_dim, latent_dim, out_node_dim, 
                 n_embeddings, commitment_cost=0.25, edge_dim=1, n_layers=4,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, tanh=False):
        super(VQEGNN, self).__init__()
        self.encoder = Encoder(seq_embedding_dim, hidden_dim, latent_dim, edge_dim=edge_dim, n_layers=n_layers,
                               act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh)
        self.vq_embedding = VQEmbedding(n_embeddings, latent_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, out_node_dim, edge_dim=edge_dim, n_layers=n_layers,
                               act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh)

    def forward(self, data):
        # 编码
        h, _ = self.encoder(data)
        
        # 层归一化
        h = F.layer_norm(h, h.size()[1:])
        
        # 向量量化
        hq, commitment_loss, codebook_loss = self.vq_embedding(h)
        
        # 解码（使用随机初始坐标）
        init_pos = torch.randn_like(data.pos)
        _, pos_recon = self.decoder(hq, init_pos, data.edge_index, data.edge_attr)
        
        return pos_recon, commitment_loss, codebook_loss