import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from typing import Tuple, List


from dataloader import RNADataset, preprocess_data
from model import VQEGNN

Tensor = torch.Tensor

dataset_path = './rna_database'

seq_embedding_dim = 16
hidden_dim = 64
latent_dim = 32
in_edge_dim = 1
n_codebook_embeddings = 128
output_node_dim = 16
commitment_beta = 0.25
egnn_layers = 4

tanh = True
normalize = True

batch_size = 8
lr_rate = 1e-4
epoches = 100

print_step = 50

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn_vqegnn(data_list):
    """
    Collate function for VQEGNN model.
    This function is used to preprocess the data before feeding it into the model.
    """
    return preprocess_data(data_list, atom_idx=11, device=cuda)


def find_alignment_kabsch(P: Tensor, Q: Tensor) -> Tuple[Tensor, Tensor]:
    """Find alignment using Kabsch algorithm between two sets of points P and Q.
    Args:
    P (torch.Tensor): A tensor of shape (N, 3) representing the first set of points.
    Q (torch.Tensor): A tensor of shape (N, 3) representing the second set of points.
    Returns:
    Tuple[Tensor, Tensor]: A tuple containing two tensors, where the first tensor is the rotation matrix R
    and the second tensor is the translation vector t. The rotation matrix R is a tensor of shape (3, 3)
    representing the optimal rotation between the two sets of points, and the translation vector t
    is a tensor of shape (3,) representing the optimal translation between the two sets of points.
    """
    # Shift points w.r.t centroid
    centroid_P, centroid_Q = P.mean(dim=0), Q.mean(dim=0)
    P_c, Q_c = P - centroid_P, Q - centroid_Q
    # Find rotation matrix by Kabsch algorithm
    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    # ensure right-handedness
    d = torch.sign(torch.linalg.det(V @ U.T))
    # Trick for torch.vmap
    diag_values = torch.cat(
        [
            torch.ones(1, dtype=P.dtype, device=P.device),
            torch.ones(1, dtype=P.dtype, device=P.device),
            d * torch.ones(1, dtype=P.dtype, device=P.device),
        ]
    )
    # This is only [[1,0,0],[0,1,0],[0,0,d]]
    M = torch.eye(3, dtype=P.dtype, device=P.device) * diag_values
    R = V @ M @ U.T
    # Find translation vectors
    t = centroid_Q[None, :] - (R @ centroid_P[None, :].T).T
    t = t.T
    return R, t.squeeze()

def calculate_rmsd_mask(pos: Tensor, ref: Tensor, mask: Tensor) -> Tensor:
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref, considering a mask.
    Args:
    pos (torch.Tensor): A tensor of shape (N, 3) representing the positions of the first set of points.
    ref (torch.Tensor): A tensor of shape (N, 3) representing the positions of the second set of points.
    mask (torch.Tensor): A boolean tensor of shape (N,) indicating which points to consider in the RMSD calculation.
    Returns:
    torch.Tensor: RMSD between the two sets of points, considering only the masked points.
    """
    if pos.shape[0] != ref.shape[0]:
        raise ValueError("pos and ref must have the same number of points")
    if mask.shape[0] != pos.shape[0]:
        raise ValueError("mask must have the same length as pos and ref")
    
    R, t = find_alignment_kabsch(ref[mask], pos[mask])
    ref0 = (R @ ref[mask].T).T + t
    rmsd = torch.linalg.norm(ref0 - pos[mask], dim=1).mean()
    return rmsd

def calculate_rmsd_batch(pos_batch: Tensor, ref_batch: Tensor, mask_batch: Tensor) -> Tensor:
    """
    Calculate the root mean square deviation (RMSD) for a batch of point sets.
    Args:
    pos_batch (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the first set of points.
    ref_batch (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the second set of points.
    mask_batch (torch.Tensor): A boolean tensor of shape (B, N) indicating which points to consider in the RMSD calculation.
    Returns:
    torch.Tensor: RMSD for each sample in the batch.
    """
    rmsd_list = []
    for pos, ref, mask in zip(pos_batch, ref_batch, mask_batch):
        rmsd = calculate_rmsd_mask(pos, ref, mask)
        rmsd_list.append(rmsd)
    return torch.tensor(rmsd_list, device=pos_batch.device).mean()

def main():
    # Load datasets
    train_dataset = RNADataset(dataset_path, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_vqegnn)

    val_dataset = RNADataset(dataset_path, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_vqegnn)

    test_dataset = RNADataset(dataset_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_vqegnn)

    # Initialize model and optimizer
    model = VQEGNN(seq_embedding_dim, hidden_dim, latent_dim, output_node_dim, n_codebook_embeddings,
               commitment_cost=commitment_beta, in_edge_dim=in_edge_dim, device=cuda, n_layers=egnn_layers, tanh=tanh, normalize=normalize)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    
    print("Start training...")
    train_losses = list()
    val_losses = list()
    train_rmsd_losses = list()
    val_rmsd_losses = list()
    train_codebook_losses = list()
    val_codebook_losses = list()
    train_commitment_losses = list()
    val_commitment_losses = list()
    best_val_loss = float('inf')

    for epoch in range(epoches):
        model.train()
        epoch_loss = 0.0
        epoch_rmsd_loss = 0.0
        epoch_codebook_loss = 0.0
        epoch_commitment_loss = 0.0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            seq, x, edges, edge_attr, seq_lengths, coord_mask = batch
            x_recon, commitment_loss, codebook_loss = model(seq, x, edges, edge_attr)

            # Calculate loss
            x_recon = x_recon.view(-1, 3)  # Flatten the coordinates for RMSD calculation
            rmsd_loss = calculate_rmsd_mask(x_recon, x, coord_mask)
            loss = rmsd_loss + commitment_loss * commitment_beta + codebook_loss
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_rmsd_loss += rmsd_loss.item()
            epoch_codebook_loss += codebook_loss.item()
            epoch_commitment_loss += commitment_loss.item()

        epoch_rmsd_loss /= len(train_loader)
        train_rmsd_losses.append(epoch_rmsd_loss)
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        epoch_codebook_loss /= len(train_loader)
        train_codebook_losses.append(epoch_codebook_loss)
        epoch_commitment_loss /= len(train_loader)
        train_commitment_losses.append(epoch_commitment_loss)

        print(f"Epoch [{epoch+1}/{epoches}] Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmsd_loss = 0.0
        val_codebook_loss = 0.0
        val_commitment_loss = 0.0
        with torch.no_grad():
            for j, val_batch in enumerate(val_loader):
                seq, x, edges, edge_attr, seq_lengths, coord_mask = val_batch
                x_recon, commitment_loss, codebook_loss = model(seq, x, edges, edge_attr)

                # Calculate loss
                x_recon = x_recon.view(-1, 3)
                rmsd_loss = calculate_rmsd_mask(x_recon, x, coord_mask)
                loss = rmsd_loss + commitment_loss * commitment_beta + codebook_loss
                
                val_loss += loss.item()
                val_rmsd_loss += rmsd_loss.item()
                val_codebook_loss += codebook_loss.item()
                val_commitment_loss += commitment_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_rmsd_loss /= len(val_loader)
        val_rmsd_losses.append(val_rmsd_loss)
        val_codebook_loss /= len(val_loader)
        val_codebook_losses.append(val_codebook_loss)
        val_commitment_loss /= len(val_loader)
        val_commitment_losses.append(val_commitment_loss)

        print(f"Epoch [{epoch+1}/{epoches}] Validation Loss: {val_loss:.4f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete.")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_codebook_losses, label='Training Codebook Loss', color='purple', linestyle='-.')
    plt.plot(val_codebook_losses, label='Validation Codebook Loss', color='brown', linestyle='--')
    plt.plot(train_commitment_losses, label='Training Commitment Loss', color='cyan', linestyle='-.')
    plt.plot(val_commitment_losses, label='Validation Commitment Loss', color='magenta', linestyle='--')
    plt.plot(train_rmsd_losses, label='Training RMSD Loss', color='green', linestyle='-.')
    plt.plot(val_rmsd_losses, label='Validation RMSD Loss', color='red', linestyle='--')
    plt.yscale('log')  # Log scale for better visibility of loss values
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses_plot.png')

    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for k, test_batch in enumerate(test_loader):
            seq, x, edges, edge_attr, seq_lengths, coord_mask = test_batch
            x_recon, commitment_loss, codebook_loss = model(seq, x, edges, edge_attr)

            # Calculate loss
            x_recon = x_recon.view(-1, 3)
            rmsd_loss = calculate_rmsd_mask(x_recon, x, coord_mask)
            loss = rmsd_loss + commitment_loss * commitment_beta + codebook_loss
            
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print("Testing complete.")

    # Calculate the average losses
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print("Average losses calculated.")
if __name__ == "__main__":
    main()