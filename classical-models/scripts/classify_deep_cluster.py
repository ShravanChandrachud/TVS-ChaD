"""
Deep Embedded Clustering (DEC)
================================
Implements unsupervised deep clustering on CLIP embeddings using the
DEC algorithm (Xie et al., 2016). Unlike LogReg and SVM, this method
does NOT use labels during training — it discovers cluster structure
from the data's geometry alone.

Three-phase pipeline:
    Phase 1 — Autoencoder Pretraining: Trains a stacked dense autoencoder
              (512→1024→512→128) to learn compressed representations via
              MSE reconstruction loss. The decoder is discarded afterward.
    Phase 2 — K-Means Initialization: Runs K-Means on the bottleneck
              representations to get initial cluster centroids.
    Phase 3 — DEC Fine-tuning: Jointly optimizes the encoder and cluster
              centroids using KL-divergence loss between soft assignments
              (Student's t-distribution) and a sharpened target distribution.
              This pushes the encoder to produce more cluster-separable
              representations.

Evaluation uses Hungarian matching to find the optimal one-to-one mapping
from discovered cluster IDs to ground-truth class labels. Reports accuracy,
NMI, ARI, F1 scores, and generates a confusion matrix and t-SNE plot.

Runs on GPU if available.
Hyperparameters are defined as constants at the top of the file.

Input:  data/embeddings/clip_vit_b32/embeddings_*.npy
Output: outputs/dec_<timestamp>/  (results JSON, confusion matrix, t-SNE plot)
        models/dec_model.pt
"""

import json
import numpy as np
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from scipy.optimize import linear_sum_assignment

from utils import (
    OUTPUTS_DIR,
    RANDOM_STATE,
    MODELS_DIR,
    load_split_embeddings,
    plot_confusion_matrix,
    plot_tsne,
    save_results,
)


N_CLUSTERS = 12
EMBEDDING_MODEL = "clip_vit_b32"

AE_EPOCHS = 100  # Autoencoder pretraining epochs
AE_LR = 1e-3  # Autoencoder learning rate
AE_BATCH_SIZE = 256

HIDDEN_1 = 1024  # First hidden layer
HIDDEN_2 = 512  # Second hidden layer
LATENT_DIM = 128

DEC_EPOCHS = 150
DEC_LR = 1e-4
DEC_BATCH_SIZE = 256
UPDATE_INTERVAL = 10
ALPHA = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Autoencoder(nn.Module):
    def __init__(self, input_dim, h1, h2, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h2, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)


class DECClusteringModule(nn.Module):
    def __init__(self, encoder, n_clusters, latent_dim, alpha=1.0):
        super().__init__()
        self.encoder = encoder
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x):
        z = self.encoder(x)
        q = self._soft_assign(z)
        return z, q

    def _soft_assign(self, z):
        dist = torch.cdist(z, self.cluster_centers, p=2).pow(2)
        q = (1.0 + dist / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q**2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p


def hungarian_match(y_true, y_pred, n_clusters):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_clusters)))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return mapping


def main():
    print("=" * 60)
    print("Deep Embedded Clustering (DEC) — CLIP Embeddings")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print(f"   N_CLUSTERS   = {N_CLUSTERS}")
    print(f"   LATENT_DIM   = {LATENT_DIM}")
    print(f"   Architecture = 512 → {HIDDEN_1} → {HIDDEN_2} → {LATENT_DIM}")
    print(f"   AE_EPOCHS    = {AE_EPOCHS}")
    print(f"   DEC_EPOCHS   = {DEC_EPOCHS}")

    print("\nLoading embeddings")
    X_train, y_train_raw, _ = load_split_embeddings("train", EMBEDDING_MODEL)
    X_test, y_test_raw, _ = load_split_embeddings("test", EMBEDDING_MODEL)
    X_val, y_val_raw, _ = load_split_embeddings("val", EMBEDDING_MODEL)

    X_all = np.vstack([X_train, X_val])
    y_all_raw = y_train_raw + y_val_raw

    unique_labels = sorted(set(y_train_raw + y_test_raw + y_val_raw))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_names = unique_labels

    y_all = np.array([label_to_idx[l] for l in y_all_raw])
    y_test = np.array([label_to_idx[l] for l in y_test_raw])

    print(f"   Train+Val: {X_all.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   Classes: {len(label_names)}")
    for name in label_names:
        count = sum(1 for l in y_all_raw if l == name)
        print(f"      {name}: {count}")

    input_dim = X_all.shape[1]

    X_all_t = torch.FloatTensor(X_all).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)

    train_dataset = TensorDataset(X_all_t)
    train_loader = DataLoader(train_dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    print("\nPHASE 1: Autoencoder Pretraining\n")

    ae = Autoencoder(input_dim, HIDDEN_1, HIDDEN_2, LATENT_DIM).to(DEVICE)
    ae_optimizer = optim.Adam(ae.parameters(), lr=AE_LR)
    ae_scheduler = optim.lr_scheduler.CosineAnnealingLR(ae_optimizer, T_max=AE_EPOCHS)
    criterion = nn.MSELoss()

    for epoch in range(AE_EPOCHS):
        ae.train()
        total_loss = 0
        for (batch,) in train_loader:
            ae_optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item() * batch.size(0)

        ae_scheduler.step()
        avg_loss = total_loss / len(X_all)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch + 1:3d}/{AE_EPOCHS} | Loss: {avg_loss:.6f} | LR: {ae_scheduler.get_last_lr()[0]:.6f}"
            )

    print("Autoencoder pretraining complete.")

    print("\nPHASE 2: K-Means Initialization\n")

    ae.eval()
    with torch.no_grad():
        z_all = ae.encode(X_all_t).cpu().numpy()

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
    kmeans_labels = kmeans.fit_predict(z_all)

    mapping_init = hungarian_match(y_all, kmeans_labels, N_CLUSTERS)
    mapped_init = np.array([mapping_init.get(c, -1) for c in kmeans_labels])
    init_acc = accuracy_score(y_all, mapped_init)
    init_nmi = normalized_mutual_info_score(y_all, kmeans_labels)
    print(f"   Initial K-Means accuracy (Hungarian): {init_acc:.4f}")
    print(f"   Initial NMI: {init_nmi:.4f}")

    print("PHASE 3: DEC Clustering Fine-tuning")

    dec = DECClusteringModule(
        encoder=ae.encoder,
        n_clusters=N_CLUSTERS,
        latent_dim=LATENT_DIM,
        alpha=ALPHA,
    ).to(DEVICE)

    dec.cluster_centers.data = torch.FloatTensor(kmeans.cluster_centers_).to(DEVICE)

    dec_optimizer = optim.Adam(dec.parameters(), lr=DEC_LR)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    dec.eval()
    with torch.no_grad():
        _, q_all = dec(X_all_t)
        p_all = DECClusteringModule.target_distribution(q_all)

    for epoch in range(DEC_EPOCHS):
        if epoch % UPDATE_INTERVAL == 0 and epoch > 0:
            dec.eval()
            with torch.no_grad():
                _, q_all = dec(X_all_t)
                p_all = DECClusteringModule.target_distribution(q_all)

        dec.train()
        total_loss = 0
        batch_idx = 0

        for start in range(0, len(X_all), DEC_BATCH_SIZE):
            end = min(start + DEC_BATCH_SIZE, len(X_all))
            batch = X_all_t[start:end]
            p_batch = p_all[start:end]

            dec_optimizer.zero_grad()
            _, q = dec(batch)

            loss = kl_loss(q.log(), p_batch)
            loss.backward()
            dec_optimizer.step()
            total_loss += loss.item() * (end - start)
            batch_idx += 1

        avg_loss = total_loss / len(X_all)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            dec.eval()
            with torch.no_grad():
                _, q_check = dec(X_all_t)
                preds_check = q_check.argmax(dim=1).cpu().numpy()

            mapping_check = hungarian_match(y_all, preds_check, N_CLUSTERS)
            mapped_check = np.array([mapping_check.get(c, -1) for c in preds_check])
            acc_check = accuracy_score(y_all, mapped_check)
            nmi_check = normalized_mutual_info_score(y_all, preds_check)

            print(
                f"   Epoch {epoch + 1:3d}/{DEC_EPOCHS} | Loss: {avg_loss:.6f} | Acc: {acc_check:.4f} | NMI: {nmi_check:.4f}"
            )

    print("DEC fine-tuning complete.")

    print("\nPHASE 4: Evaluation\n")

    dec.eval()

    with torch.no_grad():
        _, q_train = dec(X_all_t)
        preds_train = q_train.argmax(dim=1).cpu().numpy()

    mapping = hungarian_match(y_all, preds_train, N_CLUSTERS)
    mapped_train = np.array([mapping.get(c, -1) for c in preds_train])

    train_acc = accuracy_score(y_all, mapped_train)
    train_nmi = normalized_mutual_info_score(y_all, preds_train)
    train_ari = adjusted_rand_score(y_all, preds_train)

    print(f"\n   Train + Val Results:")
    print(f"      Accuracy (Hungarian): {train_acc:.4f}")
    print(f"      NMI:                  {train_nmi:.4f}")
    print(f"      ARI:                  {train_ari:.4f}")

    with torch.no_grad():
        _, q_test = dec(X_test_t)
        preds_test = q_test.argmax(dim=1).cpu().numpy()

    mapped_test = np.array([mapping.get(c, -1) for c in preds_test])
    all_labels = list(range(N_CLUSTERS))

    test_acc = accuracy_score(y_test, mapped_test)
    test_nmi = normalized_mutual_info_score(y_test, preds_test)
    test_ari = adjusted_rand_score(y_test, preds_test)
    test_f1_macro = f1_score(
        y_test, mapped_test, average="macro", labels=all_labels, zero_division=0
    )
    test_f1_weighted = f1_score(
        y_test, mapped_test, average="weighted", labels=all_labels, zero_division=0
    )

    print(f"\n   Test Results:")
    print(f"      Accuracy (Hungarian): {test_acc:.4f}")
    print(f"      F1 (macro):           {test_f1_macro:.4f}")
    print(f"      F1 (weighted):        {test_f1_weighted:.4f}")
    print(f"      NMI:                  {test_nmi:.4f}")
    print(f"      ARI:                  {test_ari:.4f}")

    report_str = classification_report(
        y_test,
        mapped_test,
        target_names=label_names,
        labels=all_labels,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        mapped_test,
        target_names=label_names,
        labels=all_labels,
        zero_division=0,
        output_dict=True,
    )
    print(f"\n{report_str}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUTS_DIR / f"dec_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, mapped_test, labels=all_labels)
    plot_confusion_matrix(
        cm,
        label_names,
        title=f"DEC Clustering [Test]\nAcc: {test_acc:.3f} | F1: {test_f1_macro:.3f} | NMI: {test_nmi:.3f}",
        save_path=run_dir / "confusion_matrix_test.png",
    )

    print("\nGenerating t-SNE visualization")
    from sklearn.preprocessing import StandardScaler

    X_test_scaled = StandardScaler().fit_transform(X_test)
    plot_tsne(
        X_test_scaled,
        y_test,
        mapped_test,
        label_names,
        f"DEC Clustering\nAcc: {test_acc:.3f} | F1: {test_f1_macro:.3f} | NMI: {test_nmi:.3f}",
        run_dir / "tsne_dec.png",
        clf_2d=None,
    )

    results = {
        "classifier": "DEC (Deep Embedded Clustering)",
        "embedding_model": EMBEDDING_MODEL,
        "hyperparameters": {
            "n_clusters": N_CLUSTERS,
            "latent_dim": LATENT_DIM,
            "hidden_layers": [HIDDEN_1, HIDDEN_2],
            "ae_epochs": AE_EPOCHS,
            "ae_lr": AE_LR,
            "dec_epochs": DEC_EPOCHS,
            "dec_lr": DEC_LR,
            "alpha": ALPHA,
            "update_interval": UPDATE_INTERVAL,
        },
        "initial_kmeans": {
            "accuracy": float(init_acc),
            "nmi": float(init_nmi),
        },
        "train_val": {
            "accuracy": float(train_acc),
            "nmi": float(train_nmi),
            "ari": float(train_ari),
        },
        "test": {
            "accuracy": float(test_acc),
            "f1_macro": float(test_f1_macro),
            "f1_weighted": float(test_f1_weighted),
            "nmi": float(test_nmi),
            "ari": float(test_ari),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        },
        "cluster_to_label_mapping": {
            str(k): label_names[v] for k, v in mapping.items()
        },
    }
    save_results(results, run_dir, "dec")

    # Save model
    torch.save(
        {
            "encoder_state_dict": ae.encoder.state_dict(),
            "dec_state_dict": dec.state_dict(),
            "cluster_centers": dec.cluster_centers.data.cpu().numpy(),
            "mapping": mapping,
        },
        MODELS_DIR / "dec_model.pt",
    )
    print(f"   Model saved: {MODELS_DIR / 'dec_model.pt'}")

    print("\nSUMMARY — DEC Deep Clustering\n")
    print(f"   Initial K-Means Acc:  {init_acc:.4f}")
    print(f"   Final Train+Val Acc:  {train_acc:.4f}")
    print(f"   Final Test Acc:       {test_acc:.4f}")
    print(f"   Test F1 (macro):      {test_f1_macro:.4f}")
    print(f"   Test NMI:             {test_nmi:.4f}")
    print(f"   Test ARI:             {test_ari:.4f}")
    print(f"\n   Cluster → Label mapping:")
    for cluster_id, label_id in sorted(mapping.items()):
        print(f"      Cluster {cluster_id:2d} → {label_names[label_id]}")
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
