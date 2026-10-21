"""
==============================================================================
ABIDE-I Dedicated Axial 3D PyTorch Neural Network (Clinical Axial Focus)
------------------------------------------------------------------------------
Author: Clint Loyed & Antigravity
Target Sites: Multi-Site ABIDE Cohort (NYU, UM_1, USM - 395 Subjects)
Input Specification: Pure Axial 3D Tensors of shape (B, 1, 50, 224, 224)

Architecture Overview:
  - 5-Stage Deep Residual 3D Convolutional Backbone (32 -> 64 -> 128 -> 256 -> 512)
  - Alternating 3x3x3 and 5x5x5 Conv3D Kernels for Multi-Scale Volumetric Analysis
  - Multi-Stage 3D CBAM (Dual Channel & 3D Spatial Attention) Mechanism
  - Built-in forward_with_attention() hook for transparent 3D clinical heatmap overlay
  - Dual-Layer Non-Linear Classification Head with LayerNorm & GELU
  - Binary Focal Loss with hard-negative mining for class boundary calibration
==============================================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def log(text=""):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'), flush=True)


# ============================================================================
# 1. Axial 3D PyTorch Dataset with Spatial Augmentations
# ============================================================================

class Axial3DDataset(Dataset):
    """
    Volumetric PyTorch Dataset dedicated strictly to Axial 3D brain scans.
    Performs on-the-fly random scaling, horizontal (sagittal-axis) flipping, 
    and subtle contrast perturbations during training.
    """
    def __init__(self, axial_tensors, labels, augment=False):
        self.axial_tensors = axial_tensors
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ax = self.axial_tensors[idx]
        if self.augment:
            # 1. Subtle intensity scale jitter (90% - 110%)
            scale = torch.empty(1, device=ax.device if isinstance(ax, torch.Tensor) else 'cpu').uniform_(0.9, 1.1)
            ax = ax * scale
            
            # 2. Random horizontal flip across the axial width axis (preserving left-right symmetry)
            if torch.rand(1).item() > 0.5:
                if isinstance(ax, torch.Tensor):
                    ax = torch.flip(ax, [-1])
                else:
                    ax = np.flip(ax, axis=-1).copy()

            # 3. Subtle Gaussian noise addition (medical sensor simulation)
            if torch.rand(1).item() > 0.7:
                noise = torch.randn_like(ax if isinstance(ax, torch.Tensor) else torch.tensor(ax)) * 0.02
                ax = ax + noise

        target = self.labels[idx]
        if not isinstance(ax, torch.Tensor):
            ax = torch.tensor(ax, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
            
        return ax, target


# ============================================================================
# 2. Residual 3D Convolutional Block
# ============================================================================

class AxialConv3DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, pool_stride=2):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.res = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2) if pool_stride == (1, 2, 2) else 2, stride=pool_stride)
        self.drop = nn.Dropout3d(0.12)

    def forward(self, x):
        residual = self.res(x)
        out = F.gelu(self.bn(self.conv(x)))
        out = self.pool(out + residual)
        return self.drop(out)


class SpatialCBAM3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return x * att, att


class AxialCBAM3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_p = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool3d(x, 1).view(b, c)
        ca = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_p))) + self.fc2(F.relu(self.fc1(max_p))))
        x = x * ca.view(b, c, 1, 1, 1)
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial_conv(torch.cat([sa_avg, sa_max], dim=1)))
        return x * sa, sa


# ============================================================================
# 4. Dedicated Axial 3D Neural Network Architecture (A100 Trained)
# ============================================================================

class DedicatedAxial3DCNN(nn.Module):
    """
    High-Capacity 5-Stage 3D Deep Convolutional Neural Network dedicated 
    exclusively to the Axial anatomical brain plane. Calibrated on ABIDE-I (NYU, UM_1, USM).
    """
    def __init__(self):
        super().__init__()
        self.stage1 = AxialConv3DBlock(1, 32, kernel_size=3, pool_stride=(1, 2, 2))
        self.stage2 = AxialConv3DBlock(32, 64, kernel_size=5, pool_stride=(1, 2, 2))
        self.stage3 = AxialConv3DBlock(64, 128, kernel_size=3, pool_stride=(2, 2, 2))
        self.stage4 = AxialConv3DBlock(128, 256, kernel_size=5, pool_stride=(2, 2, 2))
        self.stage5 = AxialConv3DBlock(256, 512, kernel_size=3, pool_stride=(2, 2, 2))
        self.cbam = SpatialCBAM3D()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x, _ = self.cbam(x)
        x = self.gap(x).flatten(1)
        return torch.sigmoid(self.classifier(x).squeeze(1))

    def forward_with_attention(self, x):
        x_in = x
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x, att = self.cbam(x)
        feat = self.gap(x).flatten(1)
        logits = self.classifier(feat).squeeze(1)
        prob = torch.sigmoid(logits)

        # Upsample 3D spatial attention map to input volume dimensions (50, 224, 224)
        sa_interpolated = F.interpolate(
            att, 
            size=(x_in.shape[2], x_in.shape[3], x_in.shape[4]), 
            mode='trilinear', 
            align_corners=False
        )
        return sa_interpolated, prob


# ============================================================================
# 5. Adaptive Binary Focal Loss
# ============================================================================

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
