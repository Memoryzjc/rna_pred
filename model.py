import torch
import torch.nn as nn
import torch.nn.functional as F

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    E(n)等变卷积层：实现对旋转、平移等几何变换的等变性
    这是EGNN的核心构建块，能够同时处理节点特征和3D坐标
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
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

    def edge_model(self, source, target, radial, edge_attr):
        """
        边模型：处理边特征，生成用于节点和坐标更新的消息
        Args:
            source: 源节点特征
            target: 目标节点特征  
            radial: 节点间距离的平方
            edge_attr: 额外的边属性
        """
        if edge_attr is None:  # 如果没有额外边属性
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)  # 计算注意力权重
            out = out * att_val          # 应用注意力机制
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        """
        节点模型：基于聚合的边信息更新节点特征
        Args:
            x: 当前节点特征
            edge_index: 边索引 [源节点, 目标节点]
            edge_attr: 边特征
            node_attr: 额外的节点属性
        """
        row, col = edge_index
        # 将边特征按目标节点聚合（每个节点收集来自所有邻居的消息）
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)  # 拼接原特征、聚合特征、额外属性
        else:
            agg = torch.cat([x, agg], dim=1)             # 拼接原特征和聚合特征
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out  # 残差连接：有助于梯度流动和稳定训练
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """
        坐标模型：更新节点坐标，保证E(n)等变性
        关键思想：x_i^{l+1} = x_i^l + Σ_j (x_i - x_j) * φ(m_ij)
        Args:
            coord: 当前节点坐标
            edge_index: 边索引
            coord_diff: 坐标差向量 (x_i - x_j)
            edge_feat: 边特征
        """
        row, col = edge_index
        # 坐标更新 = 坐标差向量 × 标量权重（保证等变性的关键）
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg  # 更新坐标
        return coord

    def coord2radial(self, edge_index, coord):
        """
        计算节点间的径向距离和坐标差向量
        Args:
            edge_index: 边索引
            coord: 节点坐标
        Returns:
            radial: 距离的平方
            coord_diff: 归一化后的坐标差向量
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]  # 计算坐标差：x_i - x_j
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)  # 距离平方：||x_i - x_j||²

        if self.normalize:
            # 归一化坐标差向量：(x_i - x_j) / ||x_i - x_j||
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        """
        E_GCL前向传播：依次执行边处理、坐标更新、节点更新
        Args:
            h: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]，包含源节点和目标节点索引
            coord: 节点坐标 [num_nodes, 3]
            edge_attr: 边属性（可选）
            node_attr: 节点属性（可选）
        Returns:
            更新后的节点特征、坐标和边属性
        """
        row, col = edge_index
        # 步骤1：计算节点间距离和坐标差
        radial, coord_diff = self.coord2radial(edge_index, coord)

        # 步骤2：生成边特征（消息传递的核心）
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        
        # 步骤3：更新坐标（等变操作，保持几何结构）
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        
        # 步骤4：更新节点特征（不变操作，提取语义信息）
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    """
    完整的E(n) Equivariant Graph Neural Network
    通过堆叠多个E_GCL层构建深层网络，用于图级别的预测任务
    """
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        '''
        Args:
        :param in_node_nf: 输入节点特征维度
        :param hidden_nf: 隐藏层特征维度
        :param out_node_nf: 输出节点特征维度
        :param in_edge_nf: 输入边特征维度
        :param device: 计算设备 (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: 激活函数
        :param n_layers: EGNN层数
        :param residual: 是否使用残差连接，建议保持True
        :param attention: 是否使用注意力机制
        :param normalize: 是否归一化坐标消息：
                    普通版本: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)φ_x(m_ij)
                    归一化版本: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)φ_x(m_ij)/||x_i - x_j||
                    可能有助于稳定性和泛化能力
        :param tanh: 是否在φ_x(m_ij)输出处使用tanh激活，可以提高稳定性但可能降低精度
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        
        # 输入嵌入层：将原始节点特征映射到隐藏维度
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        # 输出嵌入层：将隐藏特征映射到最终输出维度
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        
        # 动态添加多个E_GCL层
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        """
        EGNN前向传播：依次通过所有E_GCL层
        Args:
            h: 节点特征 [batch_size*n_nodes, node_features]
            x: 节点坐标 [batch_size*n_nodes, 3]
            edges: 边索引 [2, num_edges]
            edge_attr: 边属性
        Returns:
            更新后的节点特征和坐标
        """
        h = self.embedding_in(h)  # 输入嵌入
        # 逐层传播：每层都同时更新节点特征和坐标
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        h = self.embedding_out(h)  # 输出嵌入
        return h, x

class Encoder(nn.Module):
    """
    Encoder: Use EGNN to encode RNA sequences and coordinates.\\
    First, convert RNA sequences to embedded vectors, 
    then apply EGNN to update node features and coordinates.
    """
    def __init__(self, seq_embedding_dim, hidden_dim, out_node_dim, in_edge_dim=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(Encoder, self).__init__()
        self.seq_embedding = nn.Embedding(5, seq_embedding_dim)
        self.encoder_egnn = EGNN(seq_embedding_dim, hidden_dim, out_node_dim, in_edge_nf=in_edge_dim, 
                                 device=device, act_fn=act_fn, n_layers=n_layers,
                                 residual=residual, attention=attention, normalize=normalize, tanh=tanh)
        self.to(device)
    
    def forward(self, seq, coords, edges, edge_attr=None):
        # 将RNA序列转换为嵌入向量
        # dimensions: seq: (batch_size, max_seq_len) -> h: (batch_size, max_seq_len, seq_embedding_dim)
        h = self.seq_embedding(seq)
        h_flat = h.view(-1, h.size(-1))  # 展平为 (batch_size*n_nodes, seq_embedding_dim)
        coords_flat = coords.view(-1, coords.size(-1))  # 展平为 (batch_size*n_nodes, 3)
        
        h_flat, coords_flat = self.encoder_egnn(h_flat, coords_flat, edges, edge_attr)  
        
        h = h_flat.view(-1, seq.size(1), h_flat.size(-1))  # 恢复为 (batch_size, max_seq_len, out_node_dim)
        coords = coords_flat.view(-1, seq.size(1), coords_flat.size(-1))  # 恢复为 (batch_size, max_seq_len, 3)
        return h, coords

class VQEmbedding(nn.Module):
    """
    Vector Quantization Embedding Layer
    向量量化嵌入层：将连续特征 h 映射到离散嵌入空间 hq
    """
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, device='cpu'):
        super(VQEmbedding, self).__init__()
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)  # 初始化嵌入向量
        
        self.to(device)


    def forward(self, h):
        # encode 部分
        M, D = self.embedding.weight.size()
        h_flat = h.detach().reshape(-1, D)

        distances = torch.cdist(h_flat, self.embedding.weight, p=2) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        
        quantized = self.embedding(indices)  # (batch_size * max_seq_len, embedding_dim)
        quantized = quantized.view(*h.shape)  # (batch_size, max_seq_len, embedding_dim)

        codebook_loss = F.mse_loss(h.detach(), quantized)
        e_latent_loss = F.mse_loss(h, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = h + (quantized - h).detach()

        return quantized, commitment_loss, codebook_loss

class Decoder(nn.Module):
    """
    Decoder: Use EGNN to decode RNA sequences and coordinates from embedded vectors.
    First, use Gaussian noise as the coordinates,
    then apply EGNN to update node features and coordinates.
    """
    def __init__(self, in_node_dim, hidden_dim, out_node_dim, in_edge_dim=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        super(Decoder, self).__init__()
        self.decoder_egnn = EGNN(in_node_dim, hidden_dim, out_node_dim, in_edge_nf=in_edge_dim,
                                 device=device, act_fn=act_fn, n_layers=n_layers,
                                 residual=residual, attention=attention, normalize=normalize, tanh=tanh)
        self.to(device)

    def forward(self, z, coords, edges, edge_attr=None):
        batch_size, max_seq_len = z.size(0), z.size(1)
        z_flat = z.view(-1, z.size(-1))  # 展平为 (batch_size*max_seq_len, in_node_dim)
        coords_flat = coords.view(-1, coords.size(-1))  # 展平为 (batch_size*max_seq_len, 3)

        z_flat, coords_flat = self.decoder_egnn(z_flat, coords_flat, edges, edge_attr=edge_attr)

        z = z_flat.view(-1, max_seq_len, z_flat.size(-1))  # 恢复为 (batch_size, max_seq_len, out_node_dim)
        coords = coords_flat.view(-1, max_seq_len, coords_flat.size(-1))  # 恢复为 (batch_size, max_seq_len, 3)

        return z, coords

class VQEGNN(nn.Module):
    """
    VQEGNN: Vector Quantization + E(n) Equivariant Graph Neural Network
    结合向量量化和EGNN的完整模型，用于RNA序列的编码和解码
    """
    def __init__(self, seq_embedding_dim, hidden_dim, latent_dim, out_node_dim, 
                 n_embeddings, commitment_cost=0.25, 
                 in_edge_dim=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, 
                 residual=True, attention=False, normalize=False, tanh=False):
        super(VQEGNN, self).__init__()
        self.encoder = Encoder(seq_embedding_dim, hidden_dim, latent_dim, 
                               in_edge_dim=in_edge_dim, device=device, 
                               act_fn=act_fn, n_layers=n_layers,
                               residual=residual, attention=attention, 
                               normalize=normalize, tanh=tanh)

        self.vq_embedding = VQEmbedding(n_embeddings, latent_dim, 
                                        commitment_cost=commitment_cost, device=device)

        self.decoder = Decoder(latent_dim, hidden_dim, out_node_dim, 
                               in_edge_dim=in_edge_dim, device=device, 
                               act_fn=act_fn, n_layers=n_layers,
                               residual=residual, attention=attention, 
                               normalize=normalize, tanh=tanh)
    
    def forward(self, seq, coords, edges, edge_attr=None):
        # encoder
        # dimensions: seq: (batch_size, max_seq_len) -> h: (batch_size, max_seq_len, latent_dim)
        h, _ = self.encoder(seq, coords, edges, edge_attr)

        # vector quantization
        # dimensions: h: (batch_size, max_seq_len, latent_dim) -> hq: (batch_size, max_seq_len, latent_dim)
        hq, commitment_loss, codebook_loss = self.vq_embedding(h)

        # decoder
        # dimensions: hq: (batch_size, max_seq_len, latent_dim) -> coords: (batch_size, max_seq_len, 3)
        noise_coords = torch.randn_like(coords)  # 使用随机噪声作为初始坐标
        _, coords = self.decoder(hq, noise_coords, edges, edge_attr)

        return coords, commitment_loss, codebook_loss


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    不排序的分段求和：按segment_ids将data分组求和
    Args:
        data: 要聚合的数据 [num_elements, feature_dim]
        segment_ids: 分组标识 [num_elements]
        num_segments: 总分组数
    Returns:
        聚合后的结果 [num_segments, feature_dim]
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # 初始化零张量
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)  # 按索引累加
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """
    不排序的分段求均值：按segment_ids将data分组求平均
    Args:
        data: 要聚合的数据 [num_elements, feature_dim]
        segment_ids: 分组标识 [num_elements]
        num_segments: 总分组数
    Returns:
        聚合后的结果 [num_segments, feature_dim]
    """
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # 初始化零张量
    count = data.new_full(result_shape, 0)   # 计数张量
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)  # 避免除零错误