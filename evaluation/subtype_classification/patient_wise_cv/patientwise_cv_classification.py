#!/usr/bin/env python3
"""
Patient-Wise 5-Fold Cross-Validation Seizure Subtype Classification
====================================================================
Phase-1 subtype training with RL-augmented objective (4-class).

What this script does:
- Patient-wise 5-fold CV (no patient leakage between folds).
- Train on 80% patients and validate on 20% patients.
- Uses supervised weighted cross-entropy + policy-gradient RL auxiliary loss.

Dataset: THUSZ (4 seizure subtypes)
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

# ================================================================
# 1. CONFIGURATION
# ================================================================
BASE = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/Subtypes_LOO")
DIFFUSION_CKPT = "/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth"
NORM_STATS_DIR = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization")

CANDIDATE_TOP5_SUBTYPES = [1, 2, 7, 5, 8]

NUM_FOLDS = 5
RANDOM_SEED = 42

BATCH_SIZE = 64
NUM_EPOCHS = 60
PATIENCE = 20
LR_HEAD = 5e-4
LR_BACKBONE = 1e-5  # Lower LR for unfrozen backbone layers
UNFREEZE_LAYERS = ["bottleneck_blocks", "down_blocks.6", "down_blocks.5"]  # Last down_block + bottleneck
NUM_WORKERS = 4

# Reinforcement-learning style auxiliary loss.
RL_CONFIG = dict(
    enabled=True,
    weight=0.15,
    warmup_epochs=5,
    baseline_momentum=0.9,
    entropy_bonus=0.01,
)

ARCH = dict(
    in_channels=22,
    model_channels=32,
    channel_multipliers=[1, 2, 4, 8],
    num_res_blocks=2,
    time_emb_dim=512,
    dropout=0.1,
    attention_heads=8,
)


# ================================================================
# 2. ARCHITECTURE
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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


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
        bsz, _c, seq_len = x.shape
        h = self.conv1(F.silu(self.norm1(x)))
        t_emb = self.time_mlp(time_emb).reshape(bsz, -1, 1)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        if self.use_attention:
            an = self.attn_norm(h)
            q = self.attn_q(an).view(bsz, self.attention_heads, self.head_dim, seq_len)
            k = self.attn_k(an).view(bsz, self.attention_heads, self.head_dim, seq_len)
            v = self.attn_v(an).view(bsz, self.attention_heads, self.head_dim, seq_len)
            w = F.softmax(torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim), dim=-1)
            h = h + self.attn_proj(torch.einsum("bhqk,bhvd->bhqd", w, v).reshape(bsz, -1, seq_len))

        return h + self.residual_conv(x)


class DeepEnhancedEEGDiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels=22,
        model_channels=32,
        channel_multipliers=None,
        num_res_blocks=2,
        time_emb_dim=512,
        dropout=0.1,
        attention_heads=8,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]

        self.model_channels = model_channels
        self.channel_multipliers = channel_multipliers

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.init_conv = nn.Conv1d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        curr = model_channels
        for i, mult in enumerate(channel_multipliers):
            out = model_channels * mult
            lvl = nn.ModuleList()
            for _ in range(num_res_blocks):
                use_attn = i >= 2
                lvl.append(AdvancedResidualBlock(curr, out, time_emb_dim, dropout, use_attn, attention_heads))
                curr = out
            self.down_blocks.append(lvl)
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv1d(curr, curr, 3, stride=2, padding=1)]))

        self.bottleneck_blocks = nn.ModuleList(
            [AdvancedResidualBlock(curr, curr, time_emb_dim, dropout, True, attention_heads) for _ in range(3)]
        )


class SubtypeClassifier(nn.Module):
    def __init__(self, backbone, num_classes=5, feature_dim=256):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("probe_timesteps", torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))

        multi_scale_dim = backbone.model_channels * sum(backbone.channel_multipliers)
        total_feat_dim = multi_scale_dim * len(self.probe_timesteps)

        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

    def _get_feats(self, x, t):
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x.float())
        lvls = []
        for module in self.backbone.down_blocks:
            if isinstance(module[0], nn.Conv1d) and len(module) == 1:
                h = module[0](h)
            else:
                for block in module:
                    h = block(h, t_emb)
                lvls.append(F.adaptive_avg_pool1d(h, 1).squeeze(-1))
        return torch.cat(lvls, dim=1)

    def forward(self, x):
        assert x.ndim == 3, f"Expected 3D input [B, C, T], got {x.shape}"
        assert x.shape[1] == 22, f"Expected 22 channels, got {x.shape[1]}"

        bsz = x.shape[0]
        feats = torch.cat([self._get_feats(x, ts.expand(bsz)) for ts in self.probe_timesteps], dim=1)
        return self.classifier(feats)


# ================================================================
# 3. DATASET
# ================================================================
def _ensure_2d_sample(sample):
    sample = np.asarray(sample, dtype=np.float32)

    if sample.ndim == 4:
        if sample.shape[1] == 1:
            sample = sample.squeeze(1)
        elif sample.shape[2] == 1:
            sample = sample.squeeze(2)
    elif sample.ndim == 3:
        if sample.shape[0] == 1:
            sample = sample[0]
        elif sample.shape[1] == 1:
            sample = sample.squeeze(1)

    sample = np.squeeze(sample)
    if sample.ndim != 2:
        raise ValueError(f"Unexpected sample shape after squeeze: {sample.shape}")
    if sample.shape[0] != 22 and sample.shape[1] == 22:
        sample = sample.T
    if sample.shape[0] != 22:
        raise ValueError(f"Expected channel dimension 22, got {sample.shape}")
    return sample


class PatientSubtypeDataset(Dataset):
    def __init__(self, file_list, mean, std, subtype_map, include_patients=None):
        self.mean = mean.flatten()[:22].reshape(22, 1).astype(np.float32)
        self.std = std.flatten()[:22].reshape(22, 1).astype(np.float32)
        self.include_patients = set(include_patients) if include_patients is not None else None

        self.data_samples = []
        self.sample_patient_ids = []

        for d_file in file_list:
            l_file = str(d_file).replace("seizure_data_", "seizure_labels_")
            p_file = str(d_file).replace("seizure_data_", "seizure_patients_")
            if not os.path.exists(l_file):
                continue

            x_arr = np.load(d_file, mmap_mode="r")
            y_arr = np.load(l_file)
            pids = np.load(p_file) if os.path.exists(p_file) else np.array(["unknown"] * len(y_arr))

            mask = np.isin(y_arr, list(subtype_map.keys()))
            if self.include_patients is not None:
                mask = mask & np.isin(pids, list(self.include_patients))

            x_arr, y_arr, pids = x_arr[mask], y_arr[mask], pids[mask]
            if len(y_arr) == 0:
                continue

            y_arr = np.array([subtype_map[int(v)] for v in y_arr], dtype=np.int64)
            for i in range(len(y_arr)):
                self.data_samples.append((x_arr[i], y_arr[i]))
                self.sample_patient_ids.append(pids[i])

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample, label = self.data_samples[idx]
        sample = _ensure_2d_sample(sample)
        normed = (sample - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(normed).float(), torch.tensor(label, dtype=torch.long), self.sample_patient_ids[idx]


# ================================================================
# 4. HELPERS
# ================================================================
def collect_patient_to_files(data_files):
    patient_to_files = defaultdict(list)
    for f in data_files:
        p_file = str(f).replace("seizure_data_", "seizure_patients_")
        if os.path.exists(p_file):
            pids = np.unique(np.load(p_file))
            for pid in pids:
                patient_to_files[pid].append(f)
    return patient_to_files


def load_model(device, num_classes):
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
    model = SubtypeClassifier(backbone, num_classes=num_classes).to(device)

    print(f"Loading backbone weights from diffusion model: {DIFFUSION_CKPT}")
    try:
        diff_ckpt = torch.load(DIFFUSION_CKPT, map_location=device)
        diff_state = diff_ckpt.get("model_state_dict", diff_ckpt)

        current_backbone_state = model.backbone.state_dict()
        matched = {}

        for k, v in diff_state.items():
            if k in current_backbone_state and v.shape == current_backbone_state[k].shape:
                matched[k] = v

        model.backbone.load_state_dict(matched, strict=False)
        print(f"Loaded {len(matched)}/{len(current_backbone_state)} backbone weights from diffusion model")
    except Exception as exc:
        print(f"Warning: could not load diffusion weights: {exc}")
        print("Continuing with random initialization")

    return model


def get_param_groups(model, lr_head, lr_backbone, unfreeze_layers):
    """Create parameter groups with differential learning rates.
    
    - Classifier head: lr_head (higher)
    - Unfrozen backbone layers: lr_backbone (lower)
    - Frozen backbone layers: no grad
    """
    # First freeze everything in backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Selectively unfreeze specified layers
    unfrozen_backbone_params = []
    for layer_name in unfreeze_layers:
        for name, param in model.backbone.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                unfrozen_backbone_params.append(param)

    # Count frozen vs unfrozen
    n_frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    n_unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"Backbone: {n_frozen} frozen params, {n_unfrozen} unfrozen params")
    print(f"Unfreezing layers matching: {unfreeze_layers}")

    param_groups = [
        {"params": list(model.classifier.parameters()), "lr": lr_head, "name": "classifier"},
        {"params": unfrozen_backbone_params, "lr": lr_backbone, "name": "backbone_unfrozen"},
    ]

    return param_groups


def compute_batch_macro_f1(preds, labels):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return float(f1_score(labels_np, preds_np, average="macro", zero_division=0))


def count_subtype_support(data_files, include_patients, subtype_list):
    include_patients = set(include_patients)
    counts = {s: 0 for s in subtype_list}

    for d_file in data_files:
        l_file = str(d_file).replace("seizure_data_", "seizure_labels_")
        p_file = str(d_file).replace("seizure_data_", "seizure_patients_")
        if not os.path.exists(l_file):
            continue

        y_arr = np.load(l_file)
        pids = np.load(p_file) if os.path.exists(p_file) else np.array(["unknown"] * len(y_arr))

        patient_mask = np.isin(pids, list(include_patients))
        if not np.any(patient_mask):
            continue

        y_sel = y_arr[patient_mask]
        for s in subtype_list:
            counts[s] += int(np.sum(y_sel == s))

    return counts


def make_patient_folds(all_patients, n_folds=5, seed=42):
    all_patients = list(all_patients)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_patients)
    return [set(chunk.tolist()) for chunk in np.array_split(np.asarray(all_patients), n_folds)]


# ================================================================
# 5. TRAIN + VALIDATE ONLY
# ================================================================
def train_and_validate_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"RL config: {RL_CONFIG}")

    mean = np.load(NORM_STATS_DIR / "mean.npy")
    std = np.load(NORM_STATS_DIR / "std.npy")

    data_files = sorted(list(BASE.glob("**/seizure_data_batch_*.npy")))
    patient_to_files = collect_patient_to_files(data_files)
    all_patients = list(patient_to_files.keys())

    patient_folds = make_patient_folds(all_patients, n_folds=NUM_FOLDS, seed=RANDOM_SEED)

    # Keep one consistent 4-class mapping across all folds.
    subtype_counts = count_subtype_support(data_files, all_patients, CANDIDATE_TOP5_SUBTYPES)
    removed_subtype = min(subtype_counts.items(), key=lambda kv: kv[1])[0]
    active_subtypes = [s for s in CANDIDATE_TOP5_SUBTYPES if s != removed_subtype]
    id_map = {orig: new for new, orig in enumerate(active_subtypes)}
    num_classes = len(active_subtypes)

    print("\n" + "=" * 70)
    print("PHASE 1 ONLY: TRAIN + VALIDATION (5-FOLD CV)")
    print("=" * 70)
    print(f"Total patients: {len(all_patients)}")
    print(f"Number of folds: {NUM_FOLDS}")
    print(f"Candidate subtypes: {CANDIDATE_TOP5_SUBTYPES}")
    print(f"Subtype counts on all patients: {subtype_counts}")
    print(f"Removed subtype (lowest support): {removed_subtype}")
    print(f"Active 4-class subtypes: {active_subtypes}")

    output_dir = Path(f"./subtype_phase1_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "cv_patient_folds.json", "w") as f:
        json.dump(
            {
                "total_patients": len(all_patients),
                "num_folds": NUM_FOLDS,
                "folds": [sorted(list(v)) for v in patient_folds],
            },
            f,
            indent=2,
        )
    fold_summaries = []

    for fold_idx, val_patients in enumerate(patient_folds, start=1):
        train_patients = set(all_patients) - set(val_patients)
        fold_dir = output_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 70)
        print(f"Fold {fold_idx}/{NUM_FOLDS}")
        print(f"Train patients: {len(train_patients)} | Val patients: {len(val_patients)}")

        train_files = list(set(f for p in train_patients for f in patient_to_files[p]))
        val_files = list(set(f for p in val_patients for f in patient_to_files[p]))

        train_ds = PatientSubtypeDataset(train_files, mean, std, id_map, include_patients=train_patients)
        val_ds = PatientSubtypeDataset(val_files, mean, std, id_map, include_patients=val_patients)

        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
        if len(train_ds) == 0 or len(val_ds) == 0:
            print("Skipping fold due to empty train/val dataset")
            continue

        labels = [s[1] for s in train_ds.data_samples]
        class_counts = np.bincount(labels, minlength=num_classes)
        class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32, device=device)
        class_weights = class_weights / class_weights.sum() * num_classes
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        model = load_model(device, num_classes=num_classes)
        param_groups = get_param_groups(model, LR_HEAD, LR_BACKBONE, UNFREEZE_LAYERS)
        all_trainable_params = [p for g in param_groups for p in g["params"]]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        best_val_f1 = -1.0
        best_val_acc = 0.0
        best_epoch = -1
        patience_counter = 0
        rl_baseline = 0.0

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            total_ce = 0.0
            total_pg = 0.0
            total_reward = 0.0
            reward_batches = 0

            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)

                ce_loss = ce_loss_fn(logits, y)
                loss = ce_loss

                if RL_CONFIG["enabled"] and epoch >= RL_CONFIG["warmup_epochs"]:
                    dist = torch.distributions.Categorical(logits=logits)
                    sampled_actions = dist.sample()
                    log_prob = dist.log_prob(sampled_actions)
                    entropy = dist.entropy().mean()

                    reward = compute_batch_macro_f1(sampled_actions, y)
                    rl_baseline = (
                        RL_CONFIG["baseline_momentum"] * rl_baseline
                        + (1.0 - RL_CONFIG["baseline_momentum"]) * reward
                    )
                    advantage = reward - rl_baseline

                    pg_loss = -(advantage * log_prob.mean()) - RL_CONFIG["entropy_bonus"] * entropy
                    loss = ce_loss + RL_CONFIG["weight"] * pg_loss

                    total_pg += float(pg_loss.detach().item())
                    total_reward += reward
                    reward_batches += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_ce += float(ce_loss.item())

            scheduler.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for x, y, _ in val_loader:
                    logits = model(x.to(device))
                    val_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    val_labels.extend(y.numpy())

            val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
            val_acc = accuracy_score(val_labels, val_preds)

            avg_loss = total_loss / max(len(train_loader), 1)
            avg_ce = total_ce / max(len(train_loader), 1)

            if reward_batches > 0:
                avg_pg = total_pg / reward_batches
                avg_rw = total_reward / reward_batches
                print(
                    f"Fold {fold_idx} Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                    f"PG: {avg_pg:.4f} | RL-Reward(macroF1): {avg_rw:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )
            else:
                print(
                    f"Fold {fold_idx} Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "arch": ARCH,
                        "epoch": epoch,
                        "val_f1": float(val_f1),
                        "val_acc": float(val_acc),
                        "rl_config": RL_CONFIG,
                        "active_subtypes": active_subtypes,
                        "removed_subtype": removed_subtype,
                        "fold_index": fold_idx,
                        "unfreeze_layers": UNFREEZE_LAYERS,
                        "lr_head": LR_HEAD,
                        "lr_backbone": LR_BACKBONE,
                    },
                    fold_dir / "best_patient_wise_subtype_rl.pth",
                )
                print(f"  New best fold F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                    break

        fold_summary = {
            "fold": fold_idx,
            "best_val_f1": float(best_val_f1),
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "n_train_patients": len(train_patients),
            "n_val_patients": len(val_patients),
            "train_patient_ids": sorted(list(train_patients)),
            "val_patient_ids": sorted(list(val_patients)),
        }
        fold_summaries.append(fold_summary)

        with open(fold_dir / "fold_summary.json", "w") as f:
            json.dump(fold_summary, f, indent=2)

    if len(fold_summaries) == 0:
        raise RuntimeError("No folds were successfully trained")

    fold_best_f1s = [f["best_val_f1"] for f in fold_summaries]
    fold_best_accs = [f["best_val_acc"] for f in fold_summaries]

    summary = {
        "cv": {
            "num_folds": NUM_FOLDS,
            "n_completed_folds": len(fold_summaries),
            "best_val_f1_mean": float(np.mean(fold_best_f1s)),
            "best_val_f1_std": float(np.std(fold_best_f1s)),
            "best_val_acc_mean": float(np.mean(fold_best_accs)),
            "best_val_acc_std": float(np.std(fold_best_accs)),
            "fold_summaries": fold_summaries,
        },
        "num_classes": num_classes,
        "candidate_top5_subtypes": CANDIDATE_TOP5_SUBTYPES,
        "active_subtypes": active_subtypes,
        "removed_subtype": removed_subtype,
        "all_patient_subtype_counts": subtype_counts,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "lr_head": LR_HEAD,
            "lr_backbone": LR_BACKBONE,
            "unfreeze_layers": UNFREEZE_LAYERS,
            "num_workers": NUM_WORKERS,
        },
        "rl_config": RL_CONFIG,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "phase1_cv5_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING FINISHED (PHASE 1 ONLY, 5-FOLD CV)")
    print("=" * 70)
    print(f"CV best-val weighted F1 mean +- std: {np.mean(fold_best_f1s):.4f} +- {np.std(fold_best_f1s):.4f}")
    print(f"CV best-val accuracy mean +- std: {np.mean(fold_best_accs):.4f} +- {np.std(fold_best_accs):.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    train_and_validate_only()
