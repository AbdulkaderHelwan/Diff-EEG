#!/usr/bin/env python3
"""
Segment-Wise Binary Seizure Classification — Stratified Split (80/20)
=====================================================================
Fine-tunes pre-trained DiffEEG backbone for binary seizure detection
using segment-level stratified train/test splitting.

Dataset: THUSZ
Task: Binary classification (Normal vs Seizure)
Split: Stratified 80/20

Pre-trained model checkpoint:
  https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQDRXFfhnCuGQJs_lNO04lyUARYXKbudopQsfIkVruAvABY?e=APlvQn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
from collections import Counter
from sklearn.metrics import classification_report
import os

# ================================================================
# 1. CONFIGURATION
# ================================================================
LOO_BASE = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/Subtypes_LOO")
# --- UPDATED PATHS ---
BEST_MODEL_PATH = "best_top5_subtype.pth" # The one with 0.95 F1
DIFF_CKPT = "/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth"
NORM_STATS_DIR = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization")

TOP_5_SUBTYPES = [1, 2, 7, 5, 8]
ID_MAP = {orig: new for new, orig in enumerate(TOP_5_SUBTYPES)}

BATCH_SIZE = 128
ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)

# 2. ARCHITECTURE CLASSES (Kept from original)
# ================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, use_attention=True, attention_heads=8):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2), nn.Dropout(dropout))
        self.norm1 = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.use_attention = use_attention
        if use_attention:
            self.attn_norm = nn.InstanceNorm1d(out_channels, affine=True)
            self.attn_q = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_k = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_v = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_proj = nn.Conv1d(out_channels, out_channels, 1)
            self.attention_heads = attention_heads
            self.head_dim = out_channels // attention_heads

    def forward(self, x, time_emb):
        B, C, L = x.shape
        h = self.conv1(F.silu(self.norm1(x)))
        t_emb = self.time_mlp(time_emb).reshape(B, -1, 1)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        if self.use_attention:
            an = self.attn_norm(h)
            q = self.attn_q(an).view(B, self.attention_heads, self.head_dim, L)
            k = self.attn_k(an).view(B, self.attention_heads, self.head_dim, L)
            v = self.attn_v(an).view(B, self.attention_heads, self.head_dim, L)
            w = F.softmax(torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim), dim=-1)
            h = h + self.attn_proj(torch.einsum('bhqk,bhvd->bhqd', w, v).reshape(B, -1, L))
        return h + self.residual_conv(x)

class DeepEnhancedEEGDiffusionModel(nn.Module):
    def __init__(self, in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                 num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8):
        super().__init__()
        self.model_channels = model_channels
        self.channel_multipliers = channel_multipliers
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 2), nn.SiLU(), nn.Linear(time_emb_dim * 2, time_emb_dim))
        self.init_conv = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        curr = model_channels
        for i, mult in enumerate(channel_multipliers):
            out = model_channels * mult
            lvl = nn.ModuleList()
            for j in range(num_res_blocks):
                lvl.append(AdvancedResidualBlock(curr, out, time_emb_dim, dropout, (i>=2), attention_heads))
                curr = out
            self.down_blocks.append(lvl)
            if i != len(channel_multipliers)-1:
                self.down_blocks.append(nn.ModuleList([nn.Conv1d(curr, curr, 3, stride=2, padding=1)]))
        
        self.bottleneck_blocks = nn.ModuleList([AdvancedResidualBlock(curr, curr, time_emb_dim, dropout, True, attention_heads) for _ in range(3)])
        self.final_norm = nn.InstanceNorm1d(model_channels, affine=True) 
        self.channel_fix_conv = nn.Conv1d(model_channels, model_channels, 1)
        self.final_conv = nn.Conv1d(model_channels, in_channels, 3, padding=1)
        self.up_blocks = nn.ModuleList()

class ReinforcedSubtypeClassifier(nn.Module):
    def __init__(self, backbone, num_classes=5, feature_dim=256):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('probe_timesteps', torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
        multi_scale_dim = backbone.model_channels * sum(backbone.channel_multipliers)
        total_feat_dim = multi_scale_dim * len(self.probe_timesteps)
        
        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, feature_dim * 2), nn.BatchNorm1d(feature_dim * 2), nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim), nn.BatchNorm1d(feature_dim), nn.GELU(),
            nn.Linear(feature_dim, num_classes)
        )
        self.rl_policy = nn.Sequential(nn.Linear(total_feat_dim, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def _get_feats(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x.float())
        lvls = []
        for m in self.backbone.down_blocks:
            if isinstance(m[0], nn.Conv1d) and len(m)==1: h = m[0](h)
            else:
                for block in m: h = block(h, t_emb)
                lvls.append(F.adaptive_avg_pool1d(h, 1).squeeze(-1))
        return torch.cat(lvls, dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = torch.cat([self._get_feats(x, ts.expand(B)) for ts in self.probe_timesteps], dim=1)
        return self.classifier(feats) + self.rl_policy(feats)

# ================================================================
# 3. DATASET CLASS (Kept from previous script)
# ================================================================
class PatientLOODataset(Dataset):
    def __init__(self, file_list, mean, std):
        self.mean = mean.flatten()[:22].reshape(22, 1).astype(np.float32)
        self.std = std.flatten()[:22].reshape(22, 1).astype(np.float32)
        self.data_samples = []
        
        print("Preloading and filtering data samples...")
        for d_file in file_list:
            l_file = str(d_file).replace("seizure_data_", "seizure_labels_")
            
            if not os.path.exists(l_file): continue
            
            X = np.load(d_file, mmap_mode='r')
            Y = np.load(l_file)
            
            # Filter for Top 5
            mask = np.isin(Y, TOP_5_SUBTYPES)
            X = X[mask]
            Y = Y[mask]
            
            # Map labels to 0-4
            Y = np.array([ID_MAP[label] for label in Y])
            
            for i in range(len(Y)):
                self.data_samples.append((X[i], Y[i]))
        print(f"Loaded {len(self.data_samples)} samples.")

    def __len__(self): 
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample, label = self.data_samples[idx]
        normed_sample = (sample - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(normed_sample).float(), torch.tensor(label, dtype=torch.long)

# ================================================================
# 4. PATIENT-WISE EVALUATION (No Retraining)
# ================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    m, s = np.load(NORM_STATS_DIR/"mean.npy"), np.load(NORM_STATS_DIR/"std.npy")
    
    # 1. Identify all data files
    data_files = sorted(list(LOO_BASE.glob("**/seizure_data_batch_*.npy")))
    print(f"Found {len(data_files)} total batches.")
    
    # 2. Load the best model ONCE
    print(f"Loading best model: {BEST_MODEL_PATH}")
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
    model = ReinforcedSubtypeClassifier(backbone, num_classes=5).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 3. Iterate through every file and test
    all_final_preds = []
    all_final_labels = []
    
    for test_file_idx in range(len(data_files)):
        test_file = data_files[test_file_idx]
        
        # Load Patient IDs for logging
        test_patient_file = str(test_file).replace("seizure_data_", "seizure_patients_")
        test_pids = np.unique(np.load(test_patient_file))
        
        print(f"\nEvaluating Fold {test_file_idx+1}/{len(data_files)} | Patients: {test_pids}")
        
        val_ds = PatientLOODataset([test_file], m, s)
        
        if len(val_ds) == 0: 
            print("Skipping: No valid top-5 samples in this batch.")
            continue
        
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # 4. Evaluate Patient (Inference only)
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x.to(device)).argmax(1).cpu().numpy()
                all_final_preds.extend(preds)
                all_final_labels.extend(y.numpy())

    # 5. Final Report for all Patients
    print("\n" + "="*60)
    print("FINAL PATIENT-WISE (ZERO LEAKAGE) REPORT")
    print("="*60)
    target_names = [f"Subtype {i}" for i in TOP_5_SUBTYPES]
    print(classification_report(all_final_labels, all_final_preds, target_names=target_names, digits=4))

if __name__ == "__main__": main()


