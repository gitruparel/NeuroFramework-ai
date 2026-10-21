"""
==============================================================================
ABIDE-I Axial-Anchored Asymmetric Multi-View 3D Network
------------------------------------------------------------------------------
Author: Clint Loyed & Antigravity
Design Principle:
  - Axial Stream is Primary: Deeper 5-stage backbone (512-dim feature embedding).
  - Coronal Stream is Contextual: 4-stage backbone (128-dim) for temporal/amygdala cues.
  - Sagittal Stream is Contextual: 4-stage backbone (128-dim) for corpus callosum cues.
  - Gated Cross-Attention Fusion: Axial queries Coronal & Sagittal context representations.
==============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abide_3d_axial_cnn_pytorch import (
    AxialConv3DBlock, 
    AxialCBAM3D, 
    DedicatedAxial3DCNN, 
    BinaryFocalLoss
)
from models.abide_3d_hierarchical_cnn_pytorch import Conv3DBlock, CBAM3D


class ContextualViewExtractor(nn.Module):
    """Lightweight 4-stage 3D feature extractor for Coronal and Sagittal planes (128-dim)."""
    def __init__(self):
        super().__init__()
        self.b1 = Conv3DBlock(1, 32, 3)
        self.b2 = Conv3DBlock(32, 64, 5)
        self.b3 = Conv3DBlock(64, 128, 3)
        self.cbam = CBAM3D(128)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.cbam(x)
        return self.gap(x).view(x.size(0), -1)


class AxialAnchoredMultiViewCNN(nn.Module):
    """
    Asymmetric Multi-View Network:
    - Primary Axial Stream (512-dim)
    - Contextual Coronal Stream (128-dim)
    - Contextual Sagittal Stream (128-dim)
    - Gated Cross-Attention Fusion: Axial attention weights over auxiliary context.
    """
    def __init__(self):
        super().__init__()
        # Primary Axial Backbone (5-Stage with CBAM at every stage)
        self.axial_net = DedicatedAxial3DCNN()
        
        # Contextual streams
        self.coronal_net = ContextualViewExtractor()
        self.sagittal_net = ContextualViewExtractor()
        
        # Gated Cross-Attention Fusion
        # Total concatenated features: 512 (Axial) + 128 (Coronal) + 128 (Sagittal) = 768-dim
        self.gate_fc = nn.Sequential(
            nn.Linear(768, 3),
            nn.Softmax(dim=-1)
        )
        
        self.fusion_fc = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, ax, cor, sag):
        f_ax, _ = self.axial_net.extract_features(ax)
        f_cor = self.coronal_net(cor)
        f_sag = self.sagittal_net(sag)
        
        concat_feats = torch.cat([f_ax, f_cor, f_sag], dim=1)
        
        # Dynamic Gating across views
        gates = self.gate_fc(concat_feats)  # (B, 3)
        w_ax = gates[:, 0:1]
        w_cor = gates[:, 1:2]
        w_sag = gates[:, 2:3]
        
        # Weighted representation fusion
        f_ax_gated = f_ax * (1.0 + w_ax)
        f_cor_gated = f_cor * w_cor
        f_sag_gated = f_sag * w_sag
        
        fused = torch.cat([f_ax_gated, f_cor_gated, f_sag_gated], dim=1)
        out = self.fusion_fc(fused)
        return torch.sigmoid(out).squeeze(-1)

    def forward_with_attention(self, ax, cor, sag):
        """Returns Axial 3D attention volume along with prediction"""
        sa_interpolated, _ = self.axial_net.forward_with_attention(ax)
        prob = self.forward(ax, cor, sag)
        return sa_interpolated, prob
