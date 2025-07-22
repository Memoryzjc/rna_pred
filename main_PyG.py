import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from dataloader_PyG import RNAPyGDataset
from model_PyG import VQEGNN
from utils import calculate_masked_rmsd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_path = './rna_database'
    batch_size = 8
    lr_rate = 1e-4
    epoches = 100
    seq_embedding_dim = 16
    hidden_dim = 64
    latent_dim = 32
    out_node_dim = 3
    n_embeddings = 128
    commitment_cost = 0.25
    edge_dim = 1
    n_layers = 4

    tanh = True
    normalize = True

    train_dataset = RNAPyGDataset(dataset_path, split='train')
    val_dataset = RNAPyGDataset(dataset_path, split='val')
    test_dataset = RNAPyGDataset(dataset_path, split='test')
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 检查数据中是否有mask
    sample_data = train_dataset[0]
    has_mask = hasattr(sample_data, 'valid_mask')
    print(f"Dataset contains valid_mask: {has_mask}")
    if has_mask:
        valid_ratio = sample_data.valid_mask.float().mean().item()
        print(f"Sample valid position ratio: {valid_ratio:.2%}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = VQEGNN(seq_embedding_dim=seq_embedding_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                   out_node_dim=out_node_dim, n_embeddings=n_embeddings, commitment_cost=commitment_cost,
                   edge_dim=edge_dim, n_layers=n_layers, tanh=tanh, normalize=normalize).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    print("Starting training...")
    print("=" * 90)

    for epoch in range(epoches):
        model.train()
        total_train_loss = 0.0
        train_rmsd_losses = []
        train_commitment_losses = []
        train_codebook_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_pos, commitment_loss, codebook_loss = model(batch)
            
            # 使用mask损失函数
            target_pos = batch.pos
            valid_masks = getattr(batch, 'valid_mask', None)
            
            if valid_masks is not None:
                # 使用mask损失计算
                rmsd_loss = calculate_masked_rmsd(pred_pos, target_pos, batch.batch, valid_masks)
                train_rmsd_losses.append(rmsd_loss.item())
                train_commitment_losses.append(commitment_loss.item())
                train_codebook_losses.append(codebook_loss.item())
            else:
                # 回退到原始方法（如果没有mask）
                rmsd_loss = calculate_masked_rmsd(pred_pos, target_pos, batch.batch, None)
                total_loss = rmsd_loss + commitment_loss + codebook_loss
                train_rmsd_losses.append(rmsd_loss.item())
                train_commitment_losses.append(commitment_loss.item())
                train_codebook_losses.append(codebook_loss.item())
            
            total_loss = rmsd_loss + commitment_loss + codebook_loss
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_rmsd = sum(train_rmsd_losses) / len(train_rmsd_losses)
        avg_train_commitment = sum(train_commitment_losses) / len(train_commitment_losses)
        avg_train_codebook = sum(train_codebook_losses) / len(train_codebook_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_pos, commitment_loss, codebook_loss = model(batch)
                
                # 使用mask损失函数进行验证
                target_pos = batch.pos
                valid_masks = getattr(batch, 'valid_mask', None)
                
                if valid_masks is not None:
                    # 使用mask损失计算
                    rmsd_loss = calculate_masked_rmsd(
                        pred_pos, target_pos, batch.batch, valid_masks
                    )
                else:
                    # 回退到原始方法（如果没有mask）
                    rmsd_loss = calculate_masked_rmsd(pred_pos, target_pos, batch.batch, None)
                    total_loss = rmsd_loss + commitment_loss + codebook_loss
                
                total_loss = rmsd_loss + commitment_loss + codebook_loss
                total_val_loss += total_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 详细的训练信息输出
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epoches}]")
            print(f"  Train - Total: {avg_train_loss:.4f}, RMSD: {avg_train_rmsd:.4f}, "
                  f"Commitment: {avg_train_commitment:.4f}, Codebook: {avg_train_codebook:.4f}")
            print(f"  Val - Total: {avg_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epoches}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_PyG.pt')
            if (epoch + 1) % 10 == 0 or epoch < 10:
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss (with Mask)')
    plt.savefig('loss_curve_PyG.png')
    plt.show()

    # Test
    model.load_state_dict(torch.load('best_model_PyG.pt'))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_pos, commitment_loss, codebook_loss = model(batch)
            
            # 使用mask损失函数进行测试
            target_pos = batch.pos
            valid_masks = getattr(batch, 'valid_mask', None)
            
            if valid_masks is not None:
                # 使用mask损失计算
                rmsd_loss = calculate_masked_rmsd(
                    pred_pos, target_pos, batch.batch, valid_masks
                )
            else:
                # 回退到原始方法（如果没有mask）
                rmsd_loss = calculate_masked_rmsd(pred_pos, target_pos, batch.batch, None)
                total_loss = rmsd_loss + commitment_loss + codebook_loss
                
            total_loss = rmsd_loss + commitment_loss + codebook_loss
            total_test_loss += total_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print("=" * 50)
    print(f"🏁 Final Test Loss: {avg_test_loss:.4f}")
    print(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
