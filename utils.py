import torch
import torch.nn.functional as F

def calculate_masked_rmsd(pred_pos, true_pos, batch_indices, valid_masks):
    """
    计算考虑mask的RMSD损失
    
    Args:
        pred_pos: 预测位置 [total_nodes, 3]
        true_pos: 真实位置 [total_nodes, 3]
        batch_indices: 批次索引 [total_nodes]
        valid_masks: 有效位置mask [total_nodes]
    
    Returns:
        rmsd: 平均RMSD
    """
    rmsds = []
    batch_size = batch_indices.max().item() + 1
    
    for i in range(batch_size):
        # 获取当前样本的节点
        batch_mask = (batch_indices == i)
        sample_pred = pred_pos[batch_mask]
        sample_true = true_pos[batch_mask]
        sample_valid = valid_masks[batch_mask]
        
        # 只使用有效位置计算RMSD
        if sample_valid.sum() < 3:  # 至少需要3个点
            continue
            
        valid_pred = sample_pred[sample_valid]
        valid_true = sample_true[sample_valid]
        
        # Kabsch算法对齐
        rmsd = kabsch_rmsd(valid_pred, valid_true)
        rmsds.append(rmsd)
    
    if len(rmsds) == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    
    return torch.stack(rmsds).mean()


def kabsch_rmsd(P, Q):
    """
    使用Kabsch算法计算两组点之间的RMSD
    """
    # 中心化坐标
    P_center = P.mean(dim=0)
    Q_center = Q.mean(dim=0)
    P_centered = P - P_center
    Q_centered = Q - Q_center
    
    # Kabsch算法
    H = Q_centered.T @ P_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 确保是正确的旋转矩阵
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 应用旋转和平移
    Q_aligned = (Q_centered @ R.T) + P_center
    
    # 计算RMSD
    rmsd = torch.sqrt(F.mse_loss(P, Q_aligned))
    return rmsd
