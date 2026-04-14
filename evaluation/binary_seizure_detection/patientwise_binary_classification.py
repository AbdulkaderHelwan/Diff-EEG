#!/usr/bin/env python3
"""
Patient-Wise Binary Seizure Classification using DiffEEG Backbone
=================================================================
Fine-tunes pre-trained diffusion model for binary seizure detection
with reinforcement-assisted decision refinement.

Dataset: THUSZ (patient-wise train/dev/eval splits)
Task: Binary classification (Normal vs Seizure)

Pre-trained model checkpoint:
  https://lauedu74602-my.sharepoint.com/:u:/g/personal/abedelkader_helwan_lau_edu_lb/IQCvOrCk3Us8Sry9uUmBYtChAfJxYRlFvrFMA3Emr6fE8Wo?e=uszLoB
"""

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.distributions import Bernoulli
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from tqdm import tqdm
# import json
# from datetime import datetime
# from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
#                              confusion_matrix, classification_report, roc_curve)
# import warnings
# warnings.filterwarnings('ignore')

# from Diff_EEG_train import DeepEnhancedEEGDiffusionModel

# BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all"
# DIFFUSION_CHECKPOINT = "/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth"
# NORM_STATS_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization"

# ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
#             num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)

# BATCH_SIZE = 32
# NUM_EPOCHS = 100
# PATIENCE = 20

# # ================================================================
# # PURE NUMPY LOADING - NO CLASSES, NO TRICKS
# # ================================================================
# print("Loading normalization stats...")
# mean = np.load(Path(NORM_STATS_DIR) / "mean.npy")
# std = np.load(Path(NORM_STATS_DIR) / "std.npy")
# print(f"  ✓ Loaded\n")

# def normalize(data):
#     """Normalize batch (B, C, T)."""
#     m = mean.reshape(1, -1, 1)
#     s = std.reshape(1, -1, 1)
#     return ((data - m) / (s + 1e-6)).astype(np.float32)

# def load_files(directory):
#     """Load all .npy files from directory."""
#     path = Path(directory)
#     files = sorted([f for f in path.glob("*_batch_*.npy") if "_labels" not in f.name])
#     data = []
#     for f in tqdm(files, desc=f"  Loading {path.name}"):
#         batch = np.load(f, allow_pickle=False)
#         data.append(normalize(batch))
#     return np.concatenate(data, axis=0) if data else np.zeros((0, 22, 1280), dtype=np.float32)

# print("="*60)
# print("LOADING DATA")
# print("="*60)

# print("\nTrain seizure...")
# X_train_s = load_files(f"{BASE}/train-seizure_converted")
# print(f"  ✓ {len(X_train_s):,}")

# print("Train normal...")
# X_train_n = load_files(f"{BASE}/train-NS_converted")
# print(f"  ✓ {len(X_train_n):,}")

# print("Dev seizure...")
# X_dev_s = load_files(f"{BASE}/dev-seizure_converted")
# print(f"  ✓ {len(X_dev_s):,}")

# print("Dev normal...")
# X_dev_n = load_files(f"{BASE}/dev-non-seizure_converted")
# print(f"  ✓ {len(X_dev_n):,}")

# print("Eval seizure...")
# X_eval_s = load_files(f"{BASE}/eval-seizure_converted")
# print(f"  ✓ {len(X_eval_s):,}")

# print("Eval normal...")
# X_eval_n = load_files(f"{BASE}/eval-non-seizure_converted")
# print(f"  ✓ {len(X_eval_n):,}")

# # Combine
# X_train = np.concatenate([X_train_n, X_train_s], axis=0)
# y_train = np.concatenate([np.zeros(len(X_train_n), dtype=np.int64), np.ones(len(X_train_s), dtype=np.int64)])

# X_dev = np.concatenate([X_dev_n, X_dev_s], axis=0)
# y_dev = np.concatenate([np.zeros(len(X_dev_n), dtype=np.int64), np.ones(len(X_dev_s), dtype=np.int64)])

# X_eval = np.concatenate([X_eval_n, X_eval_s], axis=0)
# y_eval = np.concatenate([np.zeros(len(X_eval_n), dtype=np.int64), np.ones(len(X_eval_s), dtype=np.int64)])

# print(f"\n✓ Train: {len(X_train):,} samples")
# print(f"✓ Dev:   {len(X_dev):,} samples")
# print(f"✓ Eval:  {len(X_eval):,} samples\n")


# # Combine train and dev for finetuning
# X_finetune = np.concatenate([X_train, X_dev], axis=0)
# y_finetune = np.concatenate([y_train, y_dev], axis=0)

# # Split 5% for validation
# from sklearn.model_selection import train_test_split
# X_finetune_train, X_finetune_val, y_finetune_train, y_finetune_val = train_test_split(
#     X_finetune, y_finetune, test_size=0.05, random_state=42, stratify=y_finetune)

# # Class weights
# w_n = len(y_finetune_train) / (2.0 * np.sum(y_finetune_train == 0))
# w_s = len(y_finetune_train) / (2.0 * np.sum(y_finetune_train == 1))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Data loaders
# train_loader = DataLoader(TensorDataset(torch.from_numpy(X_finetune_train), torch.from_numpy(y_finetune_train)),
#                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# val_loader = DataLoader(TensorDataset(torch.from_numpy(X_finetune_val), torch.from_numpy(y_finetune_val)),
#                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# eval_loader = DataLoader(TensorDataset(torch.from_numpy(X_eval), torch.from_numpy(y_eval)),
#                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# # ================================================================
# # MODELS
# # ================================================================
# class EEGClassifier(nn.Module):
#     def __init__(self, backbone, feature_dim=256, num_classes=2, dropout=0.4):
#         super().__init__()
#         self.backbone = backbone
#         self.register_buffer('probe_timesteps', torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         multi_scale_per_t = backbone.model_channels * sum(backbone.channel_multipliers)
#         aggregated_dim = multi_scale_per_t * len(self.probe_timesteps)
        
#         self.classifier = nn.Sequential(
#             nn.Linear(aggregated_dim, feature_dim * 2),
#             nn.BatchNorm1d(feature_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim * 2, feature_dim),
#             nn.BatchNorm1d(feature_dim),
#             nn.GELU(),
#             nn.Dropout(dropout / 2),
#             nn.Linear(feature_dim, num_classes),
#         )

#     def _encode_at_t(self, x, t):
#         t_emb = self.backbone.time_mlp(t)
#         h = self.backbone.init_conv(x)
#         level_outputs = []
#         for module_list in self.backbone.down_blocks:
#             if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
#                 h = module_list[0](h)
#             else:
#                 for block in module_list:
#                     if hasattr(block, 'forward') and 'time_emb' in block.forward.__code__.co_varnames:
#                         h = block(h, t_emb)
#                     else:
#                         h = block(h)
#                 level_outputs.append(self.pool(h).squeeze(-1))
#         return torch.cat(level_outputs, dim=1)

#     def forward(self, x):
#         B = x.shape[0]
#         feats = [self._encode_at_t(x, ts.expand(B)) for ts in self.probe_timesteps]
#         combined = torch.cat(feats, dim=1)
#         logits = self.classifier(combined)
#         return logits, combined


# class ReinforcedDecisionLayer(nn.Module):
#     def __init__(self, input_dim, rl_weight=0.1, momentum=0.9):
#         super().__init__()
#         self.rl_weight = rl_weight
#         self.momentum = momentum
#         hidden = max(64, input_dim // 4)
#         self.policy = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden, 1),
#         )
#         self.register_buffer('baseline', torch.tensor(0.0))

#     def forward(self, logits, features, training=False):
#         adj = self.policy(features).squeeze(-1)
#         adjusted = logits.clone()
#         adjusted[:, 1] += adj
#         adjusted[:, 0] -= adj
#         probs = torch.softmax(adjusted, dim=1)[:, 1]
        
#         if training:
#             probs = probs.clamp(1e-6, 1 - 1e-6)
#             dist = Bernoulli(probs)
#             actions = dist.sample()
#             log_probs = dist.log_prob(actions)
#             return adjusted, probs, actions, log_probs
#         return adjusted, probs, None, None

#     def update_baseline(self, reward):
#         self.baseline.mul_(self.momentum).add_((1 - self.momentum) * reward.detach())
#         return self.baseline.detach()


# def batch_f1(preds, labels, eps=1e-6):
#     p, l = preds.float(), labels.float()
#     tp = (p * l).sum()
#     fp = (p * (1 - l)).sum()
#     fn = ((1 - p) * l).sum()
#     prec = tp / (tp + fp + eps)
#     rec = tp / (tp + fn + eps)
#     return 2 * prec * rec / (prec + rec + eps)


# def train_epoch(classifier, rl_layer, loader, optimizer, scheduler, loss_fn, device):
#     classifier.train()
#     rl_layer.train()
#     classifier.backbone.eval()
    
#     total_loss, preds_all, labels_all, rewards = [], [], [], []
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
        
#         logits, features = classifier(x)
#         adj, probs, acts, lp = rl_layer(logits, features, training=True)
        
#         ce = loss_fn(adj, y)
#         reward = batch_f1(acts, y)
#         baseline = rl_layer.update_baseline(reward)
#         rl_loss = -(reward.detach() - baseline) * lp.mean()
#         loss = ce + rl_layer.rl_weight * rl_loss
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(
#             list(classifier.classifier.parameters()) + list(rl_layer.parameters()), 1.0)
#         optimizer.step()
#         if scheduler:
#             scheduler.step()
        
#         total_loss.append(loss.item())
#         rewards.append(reward.item())
#         with torch.no_grad():
#             preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
#             labels_all.extend(y.cpu().numpy())
    
#     acc = np.mean(np.array(preds_all) == np.array(labels_all))
#     return float(np.mean(total_loss)), acc, float(np.mean(rewards))


# @torch.no_grad()
# def evaluate(classifier, rl_layer, loader, loss_fn, device):
#     classifier.eval()
#     rl_layer.eval()
#     total_loss, probs_all, preds_all, labels_all = 0.0, [], [], []
    
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         logits, features = classifier(x)
#         adj, probs, _, _ = rl_layer(logits, features, training=False)
#         total_loss += loss_fn(adj, y).item()
#         probs_all.extend(probs.cpu().numpy())
#         preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
#         labels_all.extend(y.cpu().numpy())
    
#     labels = np.array(labels_all)
#     probs = np.array(probs_all)
#     acc = np.mean(np.array(preds_all) == labels)
    
#     if len(np.unique(labels)) > 1:
#         roc_auc = roc_auc_score(labels, probs)
#         prec_c, rec_c, _ = precision_recall_curve(labels, probs)
#         pr_auc = auc(rec_c, prec_c)
#     else:
#         roc_auc = pr_auc = 0.5
    
#     return total_loss / len(loader), acc, roc_auc, pr_auc, probs, labels


# # ================================================================
# # MAIN TRAINING
# # ================================================================
# print("\nLoading backbone...")
# backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
# ckpt = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
# backbone.load_state_dict(ckpt['model_state_dict'])
# backbone.eval()

# # Freeze all backbone layers by default
# for p in backbone.parameters():
#     p.requires_grad = False

# # Unfreeze last 2 bottleneck blocks
# if hasattr(backbone, 'bottleneck_blocks'):
#     for block in backbone.bottleneck_blocks[-2:]:
#         for p in block.parameters():
#             p.requires_grad = True

# # Unfreeze last 2 encoder (down_blocks) AdvancedResidualBlocks
# if hasattr(backbone, 'down_blocks'):
#     for module_list in backbone.down_blocks[-2:]:
#         for block in module_list:
#             if hasattr(block, 'parameters'):
#                 for p in block.parameters():
#                     p.requires_grad = True
# print(f"  ✓ Loaded\n")

# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_path = Path(f"./seizure_clf_{ts}")
# output_path.mkdir(exist_ok=True)

# # Build models
# multi_scale_dim = ARCH['model_channels'] * sum(ARCH['channel_multipliers'])
# agg_dim = multi_scale_dim * 5
# classifier = EEGClassifier(backbone, feature_dim=256, dropout=0.4).to(device)

# rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=0.1).to(device)

# optimizer = optim.AdamW([
#     {'params': list(classifier.classifier.parameters()), 'lr': 5e-4, 'weight_decay': 1e-4},
#     {'params': list(rl_layer.parameters()), 'lr': 5e-4, 'weight_decay': 1e-4},
# ])
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=[5e-4, 5e-4], epochs=NUM_EPOCHS,
#     steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos',
# )
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([w_n, w_s], dtype=torch.float32).to(device))

# # Training
# print("="*60)
# print("TRAINING")
# print("="*60 + "\n")

# best_dev_auc = 0.0
# patience_counter = 0
# history = {k: [] for k in ['train_loss','train_acc','train_f1','dev_loss','dev_acc','dev_auc','dev_pr_auc']}


# for epoch in range(NUM_EPOCHS):
#     tr_loss, tr_acc, tr_f1 = train_epoch(classifier, rl_layer, train_loader, optimizer, scheduler, loss_fn, device)
#     val_loss, val_acc, val_auc, val_pr, _, _ = evaluate(classifier, rl_layer, val_loader, loss_fn, device)
#     history['train_loss'].append(tr_loss)
#     history['train_acc'].append(tr_acc)
#     history['train_f1'].append(tr_f1)
#     history['dev_loss'].append(val_loss)
#     history['dev_acc'].append(val_acc)
#     history['dev_auc'].append(val_auc)
#     history['dev_pr_auc'].append(val_pr)
#     if (epoch + 1) % 5 == 0:
#         lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch+1:3d} | Train Loss:{tr_loss:.4f} Acc:{tr_acc:.3f} F1:{tr_f1:.3f} | "
#               f"Val Loss:{val_loss:.4f} Acc:{val_acc:.3f} ROC:{val_auc:.3f} PR:{val_pr:.3f} | LR:{lr:.2e}")
#     if val_auc > best_dev_auc:
#         best_dev_auc = val_auc
#         patience_counter = 0
#         torch.save({
#             'epoch': epoch,
#             'classifier': classifier.state_dict(),
#             'rl_layer': rl_layer.state_dict(),
#             'best_dev_auc': best_dev_auc,
#             'arch': ARCH,
#         }, output_path / "best_classifier.pth")
#     else:
#         patience_counter += 1
#         if patience_counter >= PATIENCE:
#             print(f"\nEarly stopping at epoch {epoch+1}")
#             break

# # Load best and evaluate
# best = torch.load(output_path / "best_classifier.pth", map_location=device)
# classifier.load_state_dict(best['classifier'])
# rl_layer.load_state_dict(best['rl_layer'])


# _, _, _, _, val_probs, val_labels = evaluate(classifier, rl_layer, val_loader, loss_fn, device)
# _, _, _, _, eval_probs, eval_labels = evaluate(classifier, rl_layer, eval_loader, loss_fn, device)

# for name, probs, labels in [("val", val_probs, val_labels), ("eval", eval_probs, eval_labels)]:
#     preds = (probs >= 0.5).astype(int)
#     roc = roc_auc_score(labels, probs)
#     pc, rc, _ = precision_recall_curve(labels, probs)
#     pr_auc = auc(rc, pc)
    
#     print(f"\n{'='*60}")
#     print(f"{name.upper()} RESULTS")
#     print(f"{'='*60}")
#     print(f"ROC-AUC: {roc:.4f}")
#     print(f"PR-AUC: {pr_auc:.4f}")
#     print(classification_report(labels, preds, target_names=['Normal','Seizure']))

# with open(output_path / "history.json", 'w') as f:
#     json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

# print(f"\n✓ All outputs saved to {output_path}/")

#!/usr/bin/env python3
"""
Fine-tune EEG Diffusion Model — ULTRA MEMORY-EFFICIENT VERSION
Uses memory-mapped files (mmap) and no caching to minimize RAM usage.
Fixed to handle [1, 22, 1280] shape properly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Bernoulli
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             classification_report)
import warnings
import gc
warnings.filterwarnings('ignore')

from Diff_EEG_train import DeepEnhancedEEGDiffusionModel

BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all"
DIFFUSION_CHECKPOINT = "/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth"
NORM_STATS_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization"

ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
            num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)

BATCH_SIZE = 256
NUM_EPOCHS = 100
PATIENCE = 20

# ================================================================
# ZERO-COPY MEMORY-MAPPED DATASET
# ================================================================
class MemmapEEGDataset(Dataset):
    """Load data using memory-mapped files - zero RAM overhead."""
    
    def __init__(self, sample_info, mean, std):
        """
        Args:
            sample_info: List of (file_path, local_idx, label) tuples
            mean, std: Normalization parameters
        """
        self.sample_info = sample_info
        self.mean = mean.reshape(1, -1, 1).astype(np.float32)
        self.std = std.reshape(1, -1, 1).astype(np.float32)
        
    def __len__(self):
        return len(self.sample_info)
    
    def __getitem__(self, idx):
        file_path, local_idx, label = self.sample_info[idx]
        
        # Load only the specific sample using mmap
        arr = np.load(file_path, mmap_mode='r')
        sample = np.array(arr[local_idx], dtype=np.float32)  # Copy to RAM
        
        # Handle different possible shapes
        # Expected final shape: [22, 1280] (channels, time)
        
        if sample.ndim == 4:
            # Shape: [1, 1, 22, 1280] -> [22, 1280]
            sample = sample.squeeze()
        elif sample.ndim == 3:
            # Most common: [1, 22, 1280] -> [22, 1280]
            if sample.shape[0] == 1:
                sample = sample[0]
            elif sample.shape[2] == 1:
                # [22, 1280, 1] -> [22, 1280]
                sample = sample[:, :, 0]
            elif sample.shape[1] == 22:
                # [batch, 22, 1280] where batch should be removed
                sample = sample[0]
            else:
                # Try generic squeeze
                sample = np.squeeze(sample)
        elif sample.ndim == 2:
            # Already correct shape [22, 1280] or needs transpose
            if sample.shape[0] != 22 and sample.shape[1] == 22:
                sample = sample.T
        elif sample.ndim == 1:
            # Flattened, reshape
            sample = sample.reshape(22, -1)
        
        # Final verification
        if sample.ndim != 2 or sample.shape[0] != 22:
            raise ValueError(f"Could not reshape sample to [22, time]. Got shape: {sample.shape} from original shape at index {local_idx} in {file_path}")
        
        # Normalize
        sample = (sample - self.mean) / (self.std + 1e-6)
        
        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)


def build_sample_index(directory, label):
    """
    Build index mapping sample_idx -> (file_path, local_idx_in_file, label).
    Does NOT load data, only reads shapes.
    """
    path = Path(directory)
    files = sorted([f for f in path.glob("*_batch_*.npy") if "_labels" not in f.name])
    
    sample_info = []
    
    for f in files:
        # Use mmap to read shape without loading data
        arr = np.load(f, mmap_mode='r')
        n_samples = arr.shape[0]
        
        for local_idx in range(n_samples):
            sample_info.append((str(f), local_idx, label))
    
    return sample_info


# ================================================================
# BUILD SAMPLE INDICES
# ================================================================
print("="*60)
print("BUILDING SAMPLE INDEX (mmap mode - ZERO RAM usage)")
print("="*60 + "\n")

print("Loading normalization stats...")
mean = np.load(Path(NORM_STATS_DIR) / "mean.npy")
std = np.load(Path(NORM_STATS_DIR) / "std.npy")
print(f"  ✓ Shape: {mean.shape}\n")

print("Indexing train seizure...")
train_s_info = build_sample_index(f"{BASE}/train-seizure_converted", 1)
print(f"  ✓ {len(train_s_info):,} samples")

print("Indexing train normal...")
train_n_info = build_sample_index(f"{BASE}/train-NS_converted", 0)
print(f"  ✓ {len(train_n_info):,} samples")

print("Indexing dev seizure...")
dev_s_info = build_sample_index(f"{BASE}/dev-seizure_converted", 1)
print(f"  ✓ {len(dev_s_info):,} samples")

print("Indexing dev normal...")
dev_n_info = build_sample_index(f"{BASE}/dev-non-seizure_converted", 0)
print(f"  ✓ {len(dev_n_info):,} samples")

print("Indexing eval seizure...")
eval_s_info = build_sample_index(f"{BASE}/eval-seizure_converted", 1)
print(f"  ✓ {len(eval_s_info):,} samples")

print("Indexing eval normal...")
eval_n_info = build_sample_index(f"{BASE}/eval-non-seizure_converted", 0)
print(f"  ✓ {len(eval_n_info):,} samples")

# Combine
train_info = train_n_info + train_s_info
dev_info = dev_n_info + dev_s_info
eval_info = eval_n_info + eval_s_info

print(f"\n✓ Train: {len(train_info):,} samples")
print(f"✓ Dev:   {len(dev_info):,} samples")
print(f"✓ Eval:  {len(eval_info):,} samples\n")

# Combine train and dev for finetuning
finetune_info = train_info + dev_info
finetune_labels = np.array([label for _, _, label in finetune_info], dtype=np.int64)

# Split 5% for validation
from sklearn.model_selection import train_test_split
indices = np.arange(len(finetune_info))
train_idx, val_idx = train_test_split(
    indices, test_size=0.05, random_state=42, stratify=finetune_labels)

train_info_split = [finetune_info[i] for i in train_idx]
val_info_split = [finetune_info[i] for i in val_idx]

# Class weights
train_labels_array = np.array([label for _, _, label in train_info_split])
w_n = len(train_labels_array) / (2.0 * np.sum(train_labels_array == 0))
w_s = len(train_labels_array) / (2.0 * np.sum(train_labels_array == 1))

print(f"Train split: {len(train_info_split):,} samples")
print(f"Val split:   {len(val_info_split):,} samples")
print(f"Class weights: Normal={w_n:.3f}, Seizure={w_s:.3f}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Create datasets
train_dataset = MemmapEEGDataset(train_info_split, mean, std)
val_dataset = MemmapEEGDataset(val_info_split, mean, std)
eval_dataset = MemmapEEGDataset(eval_info, mean, std)

# Verify data shape
print("Verifying data shapes...")
try:
    sample, label = train_dataset[0]
    print(f"  ✓ Sample shape: {sample.shape} (expected: [22, 1280])")
    print(f"  ✓ Label: {label}")
    
    # Check a few more samples
    for i in [1, 100, 1000]:
        if i < len(train_dataset):
            s, _ = train_dataset[i]
            if s.shape != sample.shape:
                print(f"  WARNING: Shape inconsistency at index {i}: {s.shape} vs {sample.shape}")
                break
    
    print("  ✓ Shape verification completed\n")
except Exception as e:
    print(f"  ERROR during shape verification: {e}")
    raise

# Custom collate function to ensure proper batch shape
def collate_fn(batch):
    """
    Custom collate to handle any remaining shape issues.
    Ensures batch has shape [B, C, T] not [B, 1, C, T]
    """
    samples, labels = zip(*batch)
    
    # Stack samples
    samples = torch.stack(samples)
    
    # Remove any extra dimensions
    while samples.ndim > 3:
        if samples.shape[1] == 1:
            samples = samples.squeeze(1)
        else:
            break
    
    labels = torch.stack(labels)
    
    return samples, labels

# Data loaders - fewer workers to reduce memory
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True, collate_fn=collate_fn)

# Verify batch shape from DataLoader
print("Verifying batch shapes from DataLoader...")
batch_x, batch_y = next(iter(train_loader))
print(f"  ✓ Batch shape: {batch_x.shape} (expected: [16, 22, 1280])")
print(f"  ✓ Labels shape: {batch_y.shape} (expected: [16])")
assert batch_x.ndim == 3, f"Batch should be 3D [B, C, T], got {batch_x.ndim}D"
assert batch_x.shape[1] == 22, f"Expected 22 channels, got {batch_x.shape[1]}"
print("  ✓ Batch verification passed\n")

# ================================================================
# MODELS
# ================================================================
class EEGClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=256, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('probe_timesteps', torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
        self.pool = nn.AdaptiveAvgPool1d(1)
        multi_scale_per_t = backbone.model_channels * sum(backbone.channel_multipliers)
        aggregated_dim = multi_scale_per_t * len(self.probe_timesteps)
        
        self.classifier = nn.Sequential(
            nn.Linear(aggregated_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feature_dim, num_classes),
        )

    def _encode_at_t(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        level_outputs = []
        for module_list in self.backbone.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                h = module_list[0](h)
            else:
                for block in module_list:
                    if hasattr(block, 'forward') and 'time_emb' in block.forward.__code__.co_varnames:
                        h = block(h, t_emb)
                    else:
                        h = block(h)
                level_outputs.append(self.pool(h).squeeze(-1))
        return torch.cat(level_outputs, dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = [self._encode_at_t(x, ts.expand(B)) for ts in self.probe_timesteps]
        combined = torch.cat(feats, dim=1)
        logits = self.classifier(combined)
        return logits, combined


class ReinforcedDecisionLayer(nn.Module):
    def __init__(self, input_dim, rl_weight=0.1, momentum=0.9):
        super().__init__()
        self.rl_weight = rl_weight
        self.momentum = momentum
        hidden = max(64, input_dim // 4)
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, logits, features, training=False):
        adj = self.policy(features).squeeze(-1)
        adjusted = logits.clone()
        adjusted[:, 1] += adj
        adjusted[:, 0] -= adj
        probs = torch.softmax(adjusted, dim=1)[:, 1]
        
        if training:
            probs = probs.clamp(1e-6, 1 - 1e-6)
            dist = Bernoulli(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return adjusted, probs, actions, log_probs
        return adjusted, probs, None, None

    def update_baseline(self, reward):
        self.baseline.mul_(self.momentum).add_((1 - self.momentum) * reward.detach())
        return self.baseline.detach()


def batch_f1(preds, labels, eps=1e-6):
    p, l = preds.float(), labels.float()
    tp = (p * l).sum()
    fp = (p * (1 - l)).sum()
    fn = ((1 - p) * l).sum()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return 2 * prec * rec / (prec + rec + eps)


def train_epoch(classifier, rl_layer, loader, optimizer, scheduler, loss_fn, device):
    classifier.train()
    rl_layer.train()
    classifier.backbone.eval()
    
    total_loss, preds_all, labels_all, rewards = [], [], [], []
    
    for x, y in tqdm(loader, desc="  Training", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        logits, features = classifier(x)
        adj, probs, acts, lp = rl_layer(logits, features, training=True)
        
        ce = loss_fn(adj, y)
        reward = batch_f1(acts, y)
        baseline = rl_layer.update_baseline(reward)
        rl_loss = -(reward.detach() - baseline) * lp.mean()
        loss = ce + rl_layer.rl_weight * rl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(classifier.classifier.parameters()) + list(rl_layer.parameters()), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss.append(loss.item())
        rewards.append(reward.item())
        with torch.no_grad():
            preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
        
        if len(total_loss) % 100 == 0:
            torch.cuda.empty_cache()
    
    acc = np.mean(np.array(preds_all) == np.array(labels_all))
    return float(np.mean(total_loss)), acc, float(np.mean(rewards))


@torch.no_grad()
def evaluate(classifier, rl_layer, loader, loss_fn, device):
    classifier.eval()
    rl_layer.eval()
    total_loss, probs_all, preds_all, labels_all = 0.0, [], [], []
    
    for x, y in tqdm(loader, desc="  Evaluating", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, features = classifier(x)
        adj, probs, _, _ = rl_layer(logits, features, training=False)
        total_loss += loss_fn(adj, y).item()
        probs_all.extend(probs.cpu().numpy())
        preds_all.extend(torch.argmax(adj, 1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    
    labels = np.array(labels_all)
    probs = np.array(probs_all)
    acc = np.mean(np.array(preds_all) == labels)
    
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, probs)
        prec_c, rec_c, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(rec_c, prec_c)
    else:
        roc_auc = pr_auc = 0.5
    
    torch.cuda.empty_cache()
    return total_loss / len(loader), acc, roc_auc, pr_auc, probs, labels


# ================================================================
# MAIN TRAINING
# ================================================================
print("Loading backbone...")
backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
ckpt = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
backbone.load_state_dict(ckpt['model_state_dict'])
backbone.eval()

# Freeze all backbone
for p in backbone.parameters():
    p.requires_grad = False

# Unfreeze last 2 bottleneck blocks
if hasattr(backbone, 'bottleneck_blocks'):
    for block in backbone.bottleneck_blocks[-2:]:
        for p in block.parameters():
            p.requires_grad = True

# Unfreeze last 2 down_blocks
if hasattr(backbone, 'down_blocks'):
    for module_list in backbone.down_blocks[-2:]:
        for block in module_list:
            if hasattr(block, 'parameters'):
                for p in block.parameters():
                    p.requires_grad = True

print(f"  ✓ Loaded\n")

del ckpt
gc.collect()
torch.cuda.empty_cache()

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = Path(f"./seizure_clf_{ts}")
output_path.mkdir(exist_ok=True)

# Build models
multi_scale_dim = ARCH['model_channels'] * sum(ARCH['channel_multipliers'])
agg_dim = multi_scale_dim * 5
classifier = EEGClassifier(backbone, feature_dim=256, dropout=0.4).to(device)
rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=0.1).to(device)

optimizer = optim.AdamW([
    {'params': list(classifier.classifier.parameters()), 'lr': 5e-4, 'weight_decay': 1e-4},
    {'params': list(rl_layer.parameters()), 'lr': 5e-4, 'weight_decay': 1e-4},
])
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[5e-4, 5e-4], epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos',
)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([w_n, w_s], dtype=torch.float32).to(device))

# Training
print("="*60)
print("TRAINING (Memory-efficient mode)")
print("="*60 + "\n")

best_dev_auc = 0.0
patience_counter = 0
history = {k: [] for k in ['train_loss','train_acc','train_f1','dev_loss','dev_acc','dev_auc','dev_pr_auc']}

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    tr_loss, tr_acc, tr_f1 = train_epoch(classifier, rl_layer, train_loader, optimizer, scheduler, loss_fn, device)
    val_loss, val_acc, val_auc, val_pr, _, _ = evaluate(classifier, rl_layer, val_loader, loss_fn, device)
    
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['train_f1'].append(tr_f1)
    history['dev_loss'].append(val_loss)
    history['dev_acc'].append(val_acc)
    history['dev_auc'].append(val_auc)
    history['dev_pr_auc'].append(val_pr)
    
    lr = optimizer.param_groups[0]['lr']
    print(f"  Train Loss:{tr_loss:.4f} Acc:{tr_acc:.3f} F1:{tr_f1:.3f} | "
          f"Val Loss:{val_loss:.4f} Acc:{val_acc:.3f} ROC:{val_auc:.3f} PR:{val_pr:.3f} | LR:{lr:.2e}")
    
    if val_auc > best_dev_auc:
        best_dev_auc = val_auc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'classifier': classifier.state_dict(),
            'rl_layer': rl_layer.state_dict(),
            'best_dev_auc': best_dev_auc,
            'arch': ARCH,
        }, output_path / "best_classifier.pth")
        print(f"  ✓ New best model saved (AUC: {best_dev_auc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    gc.collect()
    torch.cuda.empty_cache()

# Load best and evaluate
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60 + "\n")

best = torch.load(output_path / "best_classifier.pth", map_location=device)
classifier.load_state_dict(best['classifier'])
rl_layer.load_state_dict(best['rl_layer'])
del best

_, _, _, _, val_probs, val_labels = evaluate(classifier, rl_layer, val_loader, loss_fn, device)
_, _, _, _, eval_probs, eval_labels = evaluate(classifier, rl_layer, eval_loader, loss_fn, device)

for name, probs, labels in [("val", val_probs, val_labels), ("eval", eval_probs, eval_labels)]:
    preds = (probs >= 0.5).astype(int)
    roc = roc_auc_score(labels, probs)
    pc, rc, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(rc, pc)
    
    print(f"\n{'='*60}")
    print(f"{name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(classification_report(labels, preds, target_names=['Normal','Seizure']))

with open(output_path / "history.json", 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

print(f"\n✓ All outputs saved to {output_path}/")
print(f"✓ Memory-efficient training completed successfully")