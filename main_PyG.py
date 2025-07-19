import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from dataloader_PyG import RNAPyGDataset
from model_PyG import VQEGNN


def calculate_rmsd_mask(pred_pos, true_pos, batch):
    rmsds = []
    for i in range(batch.max().item() + 1):
        mask = (batch == i)
        P = pred_pos[mask]
        Q = true_pos[mask]
        if P.size(0) == 0:
            continue
        P_center = P.mean(dim=0)
        Q_center = Q.mean(dim=0)
        P_centered = P - P_center
        Q_centered = Q - Q_center
        H = Q_centered.T @ P_centered
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        Q_aligned = (Q_centered @ R.T) + P_center
        rmsd = torch.sqrt(F.mse_loss(P, Q_aligned))
        rmsds.append(rmsd.item())
    return sum(rmsds) / len(rmsds)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_dataset = RNAPyGDataset(dataset_path, split='train')
    val_dataset = RNAPyGDataset(dataset_path, split='val')
    test_dataset = RNAPyGDataset(dataset_path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = VQEGNN(seq_embedding_dim=seq_embedding_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                   out_node_dim=out_node_dim, n_embeddings=n_embeddings, commitment_cost=commitment_cost,
                   edge_dim=edge_dim, n_layers=n_layers).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epoches):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_pos, commitment_loss, codebook_loss = model(batch)
            
            target_pos = batch.pos
            rmsd_loss = calculate_rmsd_mask(pred_pos, target_pos, batch.batch)
            
            loss = rmsd_loss + commitment_loss + codebook_loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_pos, commitment_loss, codebook_loss = model(batch)
                target_pos = batch.pos
                rmsd_loss = calculate_rmsd_mask(pred_pos, target_pos, batch.batch)
                loss = rmsd_loss + commitment_loss + codebook_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epoches}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')

    # Test
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_pos, commitment_loss, codebook_loss = model(batch)
            target_pos = batch.pos
            rmsd_loss = calculate_rmsd_mask(pred_pos, target_pos, batch.batch)
            loss = rmsd_loss + commitment_loss + codebook_loss
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    main()
