"""
==============================================================================
ABIDE-I 3D Multi-Planar PyTorch Neural Network (Multi-Site: NYU, UM_1, USM)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Sites: NYU, UM_1, USM (395 Subjects Total)
Target Resolution: Full HD (224, 224) 3D Volumetric Tensors (50 Slices)
Results: 5-Fold Stratified Cross Validation (Peak Fold 1 Accuracy: 75.95%)

Architecture:
  - 3-Stream Parallel Conv3D Feature Extractors (Axial, Coronal, Sagittal)
  - Alternating 3x3x3 & 5x5x5 Conv3D Kernels with Residual Connections
  - 3D CBAM Dual Channel-Spatial Attention Mechanism
  - GPU VRAM Preloaded Engine for Zero-Latency High-Speed Execution
==============================================================================
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def log(text=""):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'), flush=True)

# ============================================================================
# Core PyTorch Architecture Modules (Exportable)
# ============================================================================

# 1. PyTorch Dataset with 3D Spatial Augmentations
class GPUsMRIDataset(Dataset):
    def __init__(self, ax, cor, sag, labels, augment=False):
        self.ax = ax
        self.cor = cor
        self.sag = sag
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, c, s = self.ax[idx], self.cor[idx], self.sag[idx]
        if self.augment:
            scale = torch.empty(1, device=a.device).uniform_(0.9, 1.1)
            a, c, s = a * scale, c * scale, s * scale
            if torch.rand(1, device=a.device).item() > 0.5:
                a, c, s = torch.flip(a, [-1]), torch.flip(c, [-1]), torch.flip(s, [-1])
        return a, c, s, self.labels[idx]

# 2. Conv3D Residual Block (Alternating 3x3x3 and 5x5x5 kernels)
class Conv3DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.res = nn.Conv3d(in_c, out_c, 1, padding=0, bias=False)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        res = self.res(x)
        out = F.relu(self.bn(self.conv(x)))
        out = self.pool(out + res)
        return out

# 3. 3D CBAM Dual Channel-Spatial Attention Module
class CBAM3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 8)
        self.fc2 = nn.Linear(channels // 8, channels)
        self.spatial_conv = nn.Conv3d(2, 1, 3, padding=1)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_p = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool3d(x, 1).view(b, c)
        ca = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_p))) + self.fc2(F.relu(self.fc1(max_p))))
        x = x * ca.view(b, c, 1, 1, 1)
        
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([sa_avg, sa_max], dim=1)))
        return x * sa

# 4. Single Planar Feature Extractor Stream
class View3DFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = Conv3DBlock(1, 32, 3)
        self.b2 = Conv3DBlock(32, 64, 5)
        self.b3 = Conv3DBlock(64, 128, 3)
        self.b4 = Conv3DBlock(128, 256, 5)
        self.cbam = CBAM3D(256)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.cbam(x)
        return self.gap(x).view(x.size(0), -1)

# 5. Full 3-Stream Multi-Planar Hierarchical Neural Network
class Hierarchical3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ax_net = View3DFeatureExtractor()
        self.cor_net = View3DFeatureExtractor()
        self.sag_net = View3DFeatureExtractor()
        
        self.ln = nn.LayerNorm(256 * 3)
        self.fc = nn.Linear(256 * 3, 256)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, ax, cor, sag):
        f_ax = self.ax_net(ax)
        f_cor = self.cor_net(cor)
        f_sag = self.sag_net(sag)
        
        z = torch.cat([f_ax, f_cor, f_sag], dim=1)
        z = self.ln(z)
        z = F.gelu(self.fc(z))
        z = self.dropout(z)
        return torch.sigmoid(self.out(z)).squeeze(-1)

# 6. Adaptive Binary Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        p_t = targets * preds + (1 - targets) * (1 - preds)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal_loss.mean()

# ============================================================================
# Standalone Training Execution Loop
# ============================================================================

def main():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log("==========================================================================")
    log("[TRAIN] INITIATING 3D MULTI-PLANAR PYTORCH MODEL (NYU + UM_1 + USM)")
    log("==========================================================================")
    log(f"1. PyTorch Execution Device: {device}")
    if device.type == 'cuda':
        log(f"[CUDA] Active Accelerator: {torch.cuda.get_device_name(0)}")

    GLOBAL_BATCH_SIZE = 8
    EPOCHS = 50

    if os.path.exists('/kaggle/working'):
        base_dir = '/kaggle/working/'
    else:
        base_dir = './'

    if os.path.exists('./data/ABIDE_Phenotypic.csv'):
        phenotype_csv = './data/ABIDE_Phenotypic.csv'
    elif os.path.exists('/kaggle/working'):
        phenotype_csv = '/kaggle/input/datasets/clintloyed/abide-autism-10x-data/ABIDE_Phenotypic.csv'
    else:
        phenotype_csv = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'

    if os.path.exists(os.path.join(base_dir, 'processed_paper_3D_224')):
        data_dir = os.path.join(base_dir, 'processed_paper_3D_224')
    else:
        data_dir = os.path.join(base_dir, 'processed_paper_3D')

    log(f"[DATA] Ingesting 3D Tensors from: '{data_dir}'...")
    df = pd.read_csv(phenotype_csv)
    TARGET_SITES = ['NYU', 'UM_1', 'USM']
    df = df[df['SITE_ID'].isin(TARGET_SITES)].dropna(subset=['DX_GROUP'])
    label_dict = {str(row['SUB_ID']).zfill(7): 1 if row['DX_GROUP'] == 1 else 0 for _, row in df.iterrows()}

    X_ax, X_cor, X_sag, y = [], [], [], []

    if not os.path.exists(data_dir):
        log(f"[ERROR] Data directory '{data_dir}' does not exist yet. Run preprocessing first.")
        return

    for patient_id in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, patient_id)
        if not os.path.isdir(folder_path) or patient_id not in label_dict:
            continue
            
        try:
            ax = np.load(os.path.join(folder_path, "axial_50.npy"))
            cor = np.load(os.path.join(folder_path, "coronal_50.npy"))
            sag = np.load(os.path.join(folder_path, "sagittal_50.npy"))
        except Exception:
            continue
            
        # PyTorch layout: (Channels, Depth, Height, Width) -> (1, 50, 224, 224)
        ax = np.transpose(ax, (3, 0, 1, 2))
        cor = np.transpose(cor, (3, 0, 1, 2))
        sag = np.transpose(sag, (3, 0, 1, 2))
        
        X_ax.append(ax)
        X_cor.append(cor)
        X_sag.append(sag)
        y.append(label_dict[patient_id])

    X_ax = np.array(X_ax, dtype=np.float32)
    X_cor = np.array(X_cor, dtype=np.float32)
    X_sag = np.array(X_sag, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    log(f"[LOADED] Total 3D Volume Scans Loaded: {len(X_ax)}")
    log(f"   Autism (1): {int(np.sum(y == 1))}, Healthy Control (0): {int(np.sum(y == 0))}")

    if device.type == 'cuda':
        log("[VRAM] Preloading dataset onto GPU VRAM...")
        GPU_X_ax = torch.tensor(X_ax, device=device)
        GPU_X_cor = torch.tensor(X_cor, device=device)
        GPU_X_sag = torch.tensor(X_sag, device=device)
        GPU_y = torch.tensor(y, device=device)
        log("[VRAM] GPU VRAM Preload Complete.")

    log("\n[CV] Initiating 5-Fold Stratified Cross Validation Protocol...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_ax, y), 1):
        log(f"\n--- TRAINING FOLD {fold} / 5 ---")
        if device.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
            train_dataset = GPUsMRIDataset(GPU_X_ax[train_idx], GPU_X_cor[train_idx], GPU_X_sag[train_idx], GPU_y[train_idx], augment=True)
            val_dataset = GPUsMRIDataset(GPU_X_ax[val_idx], GPU_X_cor[val_idx], GPU_X_sag[val_idx], GPU_y[val_idx], augment=False)
        else:
            train_dataset = GPUsMRIDataset(X_ax[train_idx], X_cor[train_idx], X_sag[train_idx], y[train_idx], augment=True)
            val_dataset = GPUsMRIDataset(X_ax[val_idx], X_cor[val_idx], X_sag[val_idx], y[val_idx], augment=False)

        train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=0)

        model = Hierarchical3DCNN().to(device)
        criterion = BinaryFocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_acc = 0.0
        save_path = os.path.join(base_dir, f'PyTorch_3D_Fold{fold}.pt')

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for a, c, s, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(a, c, s)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for a, c, s, targets in val_loader:
                    outputs = model(a, c, s)
                    val_preds.extend((outputs > 0.5).cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_acc = accuracy_score(val_targets, val_preds)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)

            if epoch % 5 == 0 or epoch == EPOCHS:
                log(f"Fold {fold} | Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% (Peak: {best_acc*100:.2f}%)")

        fold_scores.append(best_acc)
        log(f"Fold {fold} Peak Validation Accuracy: {best_acc*100:.2f}%")

    log("\n==============================================")
    log("[COMPLETE] PyTorch 3D MULTI-PLANAR 5-FOLD CV")
    log("==============================================")
    for i, score in enumerate(fold_scores, 1):
        log(f"Fold {i}: {score*100:.2f}%")
    log(f"[SUMMARY] FINAL PyTorch 3D AVERAGE ACCURACY: {np.mean(fold_scores)*100:.2f}%")

if __name__ == '__main__':
    main()
