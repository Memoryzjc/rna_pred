import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops

class EGNNLayer(MessagePassing):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGNNLayer, self).__init__()
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
            act_fn)

        # 节点特征更新网络：基于聚合的边信息更新节点特征
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

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

    def forward(self, x, pos, edge_index, edge_attr):
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.sum(coord_diff ** 2, dim=-1, keepdim=True)

        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr, coord_diff=coord_diff, dist=dist)

    def message(self, x_i, x_j, edge_attr, coord_diff, dist):
        edge_input = torch.cat([x_i, x_j, dist, edge_attr], dim=-1)
        edge_feat = self.edge_mlp(edge_input)
        coord_update = coord_diff * self.coord_mlp(edge_feat)
        return edge_feat, coord_update

    def aggregate(self, inputs, index):
        edge_feat, coord_update = inputs
        agg_feat = torch.zeros(index.max()+1, edge_feat.size(-1), device=edge_feat.device).scatter_add_(0, index.unsqueeze(-1).expand(-1, edge_feat.size(-1)), edge_feat)
        agg_coord = torch.zeros(index.max()+1, coord_update.size(-1), device=coord_update.device).scatter_add_(0, index.unsqueeze(-1).expand(-1, coord_update.size(-1)), coord_update)
        return agg_feat, agg_coord

    def update(self, aggr_out, x):
        agg_feat, agg_coord = aggr_out
        updated_x = self.node_mlp(torch.cat([x, agg_feat], dim=-1))
        return updated_x, agg_coord


class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, edge_dim=1, n_layers=4):
        super(EGNN, self).__init__()
        self.embedding_in = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_dim=edge_dim) for _ in range(n_layers)
        ])
        self.embedding_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, pos, edge_index, edge_attr):
        x = self.embedding_in(x)
        for layer in self.layers:
            x, coord_update = layer(x, pos, edge_index, edge_attr)
            pos = pos + coord_update
        x = self.embedding_out(x)
        return x, pos


class Encoder(nn.Module):
    def __init__(self, seq_embedding_dim, hidden_dim, out_node_dim, edge_dim=1, n_layers=4):
        super(Encoder, self).__init__()
        self.seq_embedding = nn.Embedding(5, seq_embedding_dim)
        self.egnn = EGNN(seq_embedding_dim, hidden_dim, out_node_dim, edge_dim=edge_dim, n_layers=n_layers)

    def forward(self, data):
        x = self.seq_embedding(data.x.view(-1))  # (num_nodes, embed_dim)
        return self.egnn(x, data.pos, data.edge_index, data.edge_attr)


class Decoder(nn.Module):
    def __init__(self, in_node_dim, hidden_dim, out_node_dim, edge_dim=1, n_layers=4):
        super(Decoder, self).__init__()
        self.egnn = EGNN(in_node_dim, hidden_dim, out_node_dim, edge_dim=edge_dim, n_layers=n_layers)

    def forward(self, z, pos, edge_index, edge_attr):
        return self.egnn(z, pos, edge_index, edge_attr)


class VQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQEmbedding, self).__init__()
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)

    def forward(self, h):
        h_flat = h.detach().reshape(-1, h.size(-1))
        distances = torch.cdist(h_flat, self.embedding.weight, p=2) ** 2
        indices = torch.argmin(distances.float(), dim=-1)
        quantized = self.embedding(indices).view(*h.shape)

        codebook_loss = F.mse_loss(h.detach(), quantized)
        e_latent_loss = F.mse_loss(h, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = h + (quantized - h).detach()
        return quantized, commitment_loss, codebook_loss


class VQEGNN(nn.Module):
    def __init__(self, seq_embedding_dim, hidden_dim, latent_dim, out_node_dim, 
                 n_embeddings, commitment_cost=0.25, edge_dim=1, n_layers=4):
        super(VQEGNN, self).__init__()
        self.encoder = Encoder(seq_embedding_dim, hidden_dim, latent_dim, edge_dim=edge_dim, n_layers=n_layers)
        self.vq_embedding = VQEmbedding(n_embeddings, latent_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, out_node_dim, edge_dim=edge_dim, n_layers=n_layers)

    def forward(self, data):
        h, _ = self.encoder(data)
        h = F.layer_norm(h, h.size()[1:])
        hq, commitment_loss, codebook_loss = self.vq_embedding(h)
        init_pos = torch.randn_like(data.pos)
        _, pos_recon = self.decoder(hq, init_pos, data.edge_index, data.edge_attr)
        return pos_recon, commitment_loss, codebook_loss
