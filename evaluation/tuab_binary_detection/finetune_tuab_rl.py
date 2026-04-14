#!/usr/bin/env python3
"""
Fine-tune DiffEEG on TUAB Dataset — Binary Classification (Normal vs Abnormal)
===============================================================================
TUAB few-shot binary classification with diffusion-backbone fine-tuning and
reinforcement-assisted decision refinement.

Dataset: Temple University Abnormal EEG Corpus (TUAB)
Task: Binary classification (normal vs abnormal)

Key points:
- Uses processed TUAB folders under `processed_tuab`
- Binary task: normal (0) vs abnormal (1)
- Runs few-shot experiments for K in {10, 50, 100, 500}
- Compares frozen vs partially unfrozen encoder modes
- Uses the predefined dataset split only:
  all `train-*` folders for training and `test-*` folders for testing
- Reports accuracy, F1, precision, sensitivity, specificity, ROC-AUC,
  PR-AUC, false alarms, and false alarm rate on the test set
"""

from __future__ import annotations

import gc
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from Diff_EEG_train import DeepEnhancedEEGDiffusionModel


BASE = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/TUAB/processed_tuab")
DIFFUSION_CHECKPOINT = Path("/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth")
NORM_STATS_DIR = BASE / "normalization"

TRAIN_DIRS = {
    0: BASE / "train-normal",
    1: BASE / "train-seizure",  # Processed from TUAB abnormal folder.
}
TEST_DIRS = {
    0: BASE / "test-normal",
    1: BASE / "test-seizure",   # Processed from TUAB abnormal folder.
}

CLASS_NAMES = ["Normal", "Abnormal"]
# High-accuracy preset: use all predefined train data and the stronger mode.
# k <= 0 means "use all available samples per class".
K_VALUES = [-1]
FREEZE_MODES = ["unfrozen"]

ARCH = dict(
    in_channels=22,
    model_channels=32,
    channel_multipliers=[1, 2, 4, 8],
    num_res_blocks=2,
    time_emb_dim=512,
    dropout=0.1,
    attention_heads=8,
)

BATCH_SIZE = 128
NUM_EPOCHS = 60
PATIENCE = 10
SEED = 42
NUM_WORKERS = 2
RL_WEIGHT = 0.02
HEAD_LR = 5e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs
RESUME_DIR = None  # Set to a previous output dir path to resume training


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MemmapEEGDataset(Dataset):
    def __init__(self, sample_info: Sequence[Tuple[str, int, int]], mean: np.ndarray, std: np.ndarray):
        self.sample_info = list(sample_info)
        self.mean = mean.reshape(-1).astype(np.float32)[:22].reshape(22, 1)
        self.std = std.reshape(-1).astype(np.float32)[:22].reshape(22, 1)

    def __len__(self) -> int:
        return len(self.sample_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, local_idx, label = self.sample_info[idx]
        arr = np.load(file_path, mmap_mode="r", allow_pickle=True)
        sample = arr[local_idx]
        sample = self._fix_shape(sample, file_path=file_path, local_idx=local_idx)
        sample = (sample - self.mean) / (self.std + 1e-6)
        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _fix_shape(sample, file_path: str = "", local_idx: int | None = None) -> np.ndarray:
        sample = np.asarray(sample)

        # Some batches may contain wrapped object/scalar entries; unwrap them.
        while sample.ndim == 0 and sample.dtype == object:
            sample = np.asarray(sample.item())

        sample = np.squeeze(sample)
        if sample.ndim == 0 and sample.size == 1:
            raise ValueError(
                f"Scalar sample encountered at idx={local_idx} from {file_path}. "
                "This batch entry is not a valid EEG segment."
            )
        if sample.ndim == 3 and sample.shape[0] == 1:
            sample = sample[0]
        if sample.ndim == 3 and sample.shape[-1] == 1:
            sample = sample[..., 0]
        if sample.ndim == 2 and sample.shape[0] != 22 and sample.shape[1] == 22:
            sample = sample.T
        if sample.ndim == 1 and sample.size % 22 == 0:
            sample = sample.reshape(22, -1)
        if sample.ndim != 2 or sample.shape[0] != 22:
            raise ValueError(
                f"Unexpected sample shape: {sample.shape} at idx={local_idx} from {file_path}"
            )
        return sample.astype(np.float32, copy=False)


def collate_fn(batch):
    samples, labels = zip(*batch)
    samples = torch.stack(samples)
    labels = torch.stack(labels)
    while samples.ndim > 3 and samples.shape[1] == 1:
        samples = samples.squeeze(1)
    return samples, labels


def build_sample_index(directory: Path, label: int) -> List[Tuple[str, int, int]]:
    files = sorted(p for p in directory.glob("*_batch_*.npy") if p.is_file() and "_labels" not in p.name)
    sample_info: List[Tuple[str, int, int]] = []
    skipped_files: List[Tuple[str, tuple, str]] = []

    for file_path in tqdm(files, desc=f"Indexing {directory.name}"):
        arr = np.load(file_path, mmap_mode="r", allow_pickle=True)
        if arr.ndim < 2 or arr.shape[0] == 0:
            skipped_files.append((file_path.name, tuple(arr.shape), str(arr.dtype)))
            continue
        if arr.ndim >= 3 and arr.shape[1] == 22:
            n_samples = int(arr.shape[0])
            sample_info.extend((str(file_path), idx, label) for idx in range(n_samples))
            continue
        # Fall back to per-sample validation for irregular arrays.
        valid_count = 0
        for idx in range(int(arr.shape[0])):
            try:
                MemmapEEGDataset._fix_shape(arr[idx], file_path=str(file_path), local_idx=idx)
                sample_info.append((str(file_path), idx, label))
                valid_count += 1
            except Exception:
                continue
        if valid_count == 0:
            skipped_files.append((file_path.name, tuple(arr.shape), str(arr.dtype)))

    if skipped_files:
        print(f"Warning: skipped {len(skipped_files)} invalid files in {directory.name}")
        for name, shape, dtype in skipped_files[:10]:
            print(f"  - {name}: shape={shape}, dtype={dtype}")
    return sample_info


def sample_few_shot(sample_info: Sequence[Tuple[str, int, int]], k: int, seed: int):
    labels = np.array([label for _, _, label in sample_info], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sampled: List[Tuple[str, int, int]] = []
    per_class: Dict[int, int] = {}

    for class_id in (0, 1):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {class_id}")
        if k <= 0:
            chosen = class_indices
            take = len(class_indices)
        else:
            take = min(k, len(class_indices))
            chosen = rng.choice(class_indices, size=take, replace=False)
        sampled.extend(sample_info[i] for i in chosen)
        per_class[class_id] = int(take)

    rng.shuffle(sampled)
    return sampled, per_class


class EEGClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int = 256, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("probe_timesteps", torch.tensor([50, 250, 500, 750, 950], dtype=torch.long))
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

    def _encode_at_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.backbone.time_mlp(t)
        h = self.backbone.init_conv(x)
        level_outputs = []
        for module_list in self.backbone.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                h = module_list[0](h)
            else:
                for block in module_list:
                    if hasattr(block, "forward") and "time_emb" in block.forward.__code__.co_varnames:
                        h = block(h, t_emb)
                    else:
                        h = block(h)
                level_outputs.append(self.pool(h).squeeze(-1))
        return torch.cat(level_outputs, dim=1)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        feats = [self._encode_at_t(x, ts.expand(batch_size)) for ts in self.probe_timesteps]
        combined = torch.cat(feats, dim=1)
        logits = self.classifier(combined)
        return logits, combined


class ReinforcedDecisionLayer(nn.Module):
    def __init__(self, input_dim: int, rl_weight: float = RL_WEIGHT, momentum: float = 0.9):
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
        self.register_buffer("baseline", torch.tensor(0.0))

    def forward(self, logits: torch.Tensor, features: torch.Tensor, training: bool = False):
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

    def update_baseline(self, reward: torch.Tensor) -> torch.Tensor:
        self.baseline.mul_(self.momentum).add_((1 - self.momentum) * reward.detach())
        return self.baseline.detach()


@dataclass
class EvalMetrics:
    accuracy: float
    f1: float
    precision: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    false_alarms: int
    false_alarm_rate: float
    false_alarm_per_sample: float
    threshold: float
    tp: int
    tn: int
    fp: int
    fn: int


def batch_f1(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    preds = preds.float()
    labels = labels.float()
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def setup_backbone(backbone: nn.Module, freeze_mode: str) -> None:
    for param in backbone.parameters():
        param.requires_grad = False

    if freeze_mode == "unfrozen":
        for param in backbone.time_mlp.parameters():
            param.requires_grad = True
        n_blocks = len(backbone.down_blocks)
        for module_list in backbone.down_blocks[max(0, n_blocks - 2):]:
            for block in module_list:
                for param in block.parameters():
                    param.requires_grad = True



def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module


def get_backbone(classifier: nn.Module) -> nn.Module:
    return unwrap_module(classifier).backbone


def create_optimizer(classifier: EEGClassifier, rl_layer: ReinforcedDecisionLayer, freeze_mode: str):
    base_classifier = unwrap_module(classifier)
    base_rl_layer = unwrap_module(rl_layer)
    if freeze_mode == "frozen":
        return optim.AdamW(
            [
                {"params": base_classifier.classifier.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
                {"params": base_rl_layer.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
            ]
        )

    backbone_params = [
        p for name, p in classifier.named_parameters()
        if p.requires_grad and "backbone" in name
    ]
    head_params = [
        p for name, p in classifier.named_parameters()
        if p.requires_grad and "backbone" not in name
    ]
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
            {"params": base_rl_layer.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ]
    )


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> EvalMetrics:
    preds = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    balanced_accuracy = 0.5 * (specificity + sensitivity)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = 0.5

    return EvalMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        precision=float(precision_score(labels, preds, zero_division=0)),
        sensitivity=float(recall_score(labels, preds, zero_division=0)),
        specificity=float(specificity),
        balanced_accuracy=float(balanced_accuracy),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        false_alarms=int(fp),
        false_alarm_rate=float(fp / (fp + tn) if (fp + tn) else 0.0),
        false_alarm_per_sample=float(fp / len(labels) if len(labels) else 0.0),
        threshold=float(threshold),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
    )



def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, EvalMetrics]:
    best_threshold = 0.5
    best_metrics = compute_metrics(labels, probs, best_threshold)
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = compute_metrics(labels, probs, float(threshold))
        current_key = (metrics.f1, metrics.balanced_accuracy, metrics.roc_auc)
        best_key = (best_metrics.f1, best_metrics.balanced_accuracy, best_metrics.roc_auc)
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics



def train_epoch(
    classifier: EEGClassifier,
    rl_layer: ReinforcedDecisionLayer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    freeze_mode: str,
):
    classifier.train()
    rl_layer.train()
    backbone = get_backbone(classifier)
    if freeze_mode == "frozen":
        backbone.eval()
    else:
        backbone.train()

    losses = []
    train_probs = []
    train_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits, features = classifier(x)
        adjusted, probs, actions, log_probs = rl_layer(logits, features, training=True)

        ce_loss = loss_fn(adjusted, y)
        reward = batch_f1(actions, y)
        baseline = unwrap_module(rl_layer).update_baseline(reward)
        rl_loss = -(reward.detach() - baseline) * log_probs.mean()
        loss = ce_loss + unwrap_module(rl_layer).rl_weight * rl_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in list(classifier.parameters()) + list(rl_layer.parameters()) if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        losses.append(loss.item())
        train_probs.extend(probs.detach().cpu().numpy())
        train_labels.extend(y.detach().cpu().numpy())

    train_metrics = compute_metrics(np.asarray(train_labels), np.asarray(train_probs), 0.5)
    return float(np.mean(losses)), train_metrics


@torch.no_grad()
def collect_predictions(
    classifier: EEGClassifier,
    rl_layer: ReinforcedDecisionLayer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    classifier.eval()
    rl_layer.eval()
    losses = []
    probs_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, features = classifier(x)
        adjusted, probs, _, _ = rl_layer(logits, features, training=False)
        losses.append(loss_fn(adjusted, y).item())
        probs_all.extend(probs.cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    return float(np.mean(losses)), np.asarray(probs_all), np.asarray(labels_all)



def make_loader(sample_info, mean, std, batch_size, shuffle):
    dataset = MemmapEEGDataset(sample_info, mean, std)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )



def summarize_counts(sample_info: Sequence[Tuple[str, int, int]]) -> Dict[str, int]:
    labels = np.array([label for _, _, label in sample_info], dtype=np.int64)
    return {
        CLASS_NAMES[0]: int((labels == 0).sum()),
        CLASS_NAMES[1]: int((labels == 1).sum()),
    }



def run_experiment(
    train_pool: Sequence[Tuple[str, int, int]],
    test_info: Sequence[Tuple[str, int, int]],
    mean: np.ndarray,
    std: np.ndarray,
    k_value: int,
    freeze_mode: str,
    device: torch.device,
    output_dir: Path,
    data_parallel_device_ids: List[int] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_train, per_class_used = sample_few_shot(
        train_pool,
        k=k_value,
        seed=SEED + k_value + (0 if freeze_mode == "frozen" else 1000),
    )

    train_loader = make_loader(sampled_train, mean, std, BATCH_SIZE, shuffle=True)
    test_loader = make_loader(test_info, mean, std, BATCH_SIZE, shuffle=False)

    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
    checkpoint = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint

    classifier = EEGClassifier(backbone=backbone, num_classes=2, dropout=0.4).to(device)
    setup_backbone(classifier.backbone, freeze_mode)

    agg_dim = ARCH["model_channels"] * sum(ARCH["channel_multipliers"]) * 5
    rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=RL_WEIGHT).to(device)

    if data_parallel_device_ids and len(data_parallel_device_ids) > 1:
        classifier = nn.DataParallel(classifier, device_ids=data_parallel_device_ids)
        rl_layer = nn.DataParallel(rl_layer, device_ids=data_parallel_device_ids)

    optimizer = create_optimizer(classifier, rl_layer, freeze_mode)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    train_labels = np.array([label for _, _, label in sampled_train], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.sum() * 2
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_score = (-math.inf, -math.inf, -math.inf)
    best_state = None
    best_train_metrics = None
    patience_counter = 0
    history = []
    start_epoch = 1

    # ── Resume from checkpoint if available ──
    k_tag = "all" if k_value <= 0 else str(k_value)
    run_name = f"k{k_tag}_{freeze_mode}"
    ckpt_path = output_dir / f"{run_name}_checkpoint.pth"
    best_path = output_dir / f"{run_name}_best.pth"

    if ckpt_path.exists():
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        unwrap_module(classifier).load_state_dict(ckpt["classifier"])
        unwrap_module(rl_layer).load_state_dict(ckpt["rl_layer"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_score = tuple(ckpt["best_score"])
        patience_counter = ckpt["patience_counter"]
        history = ckpt.get("history", [])
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device)
            best_state = {
                "classifier": best_ckpt["classifier"],
                "rl_layer": best_ckpt["rl_layer"],
                "epoch": best_ckpt["best_epoch"],
                "threshold": best_ckpt.get("threshold", 0.5),
            }
            best_train_metrics = EvalMetrics(**best_ckpt["best_train_metrics"]) if "best_train_metrics" in best_ckpt else None
            del best_ckpt
        del ckpt
        print(f"  Resumed at epoch {start_epoch}, best_score={best_score}, patience={patience_counter}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_loss, train_metrics = train_epoch(
            classifier, rl_layer, train_loader, optimizer, loss_fn, device, freeze_mode
        )
        scheduler.step()

        # Use train metrics for checkpoint selection so all predefined train data
        # remains available for fitting (no holdout split from train folders).
        score = (train_metrics.f1, train_metrics.balanced_accuracy, train_metrics.roc_auc)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": asdict(train_metrics),
        })

        print(
            f"  Epoch {epoch:02d} | "
            f"Train F1 {train_metrics.f1:.4f} | "
            f"Train Acc {train_metrics.accuracy:.4f} | "
            f"Train ROC {train_metrics.roc_auc:.4f}"
        )

        if score > best_score:
            best_score = score
            best_train_metrics = deepcopy(train_metrics)
            best_state = {
                "classifier": deepcopy(unwrap_module(classifier).state_dict()),
                "rl_layer": deepcopy(unwrap_module(rl_layer).state_dict()),
                "epoch": epoch,
                "threshold": 0.5,
            }
            patience_counter = 0
            # Save best model to disk immediately
            torch.save(
                {
                    "classifier": best_state["classifier"],
                    "rl_layer": best_state["rl_layer"],
                    "best_epoch": epoch,
                    "threshold": 0.5,
                    "best_score": list(best_score),
                    "best_train_metrics": asdict(best_train_metrics),
                    "freeze_mode": freeze_mode,
                    "k_value": k_value,
                },
                best_path,
            )
            print(f"  >> New best model saved (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Periodic checkpoint for crash/timeout recovery
        if epoch % CHECKPOINT_EVERY == 0 or epoch == NUM_EPOCHS:
            torch.save(
                {
                    "classifier": unwrap_module(classifier).state_dict(),
                    "rl_layer": unwrap_module(rl_layer).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_score": list(best_score),
                    "patience_counter": patience_counter,
                    "history": history,
                },
                ckpt_path,
            )
            print(f"  >> Checkpoint saved (epoch {epoch})")

    if best_state is None:
        # Check if a best model was saved to disk from a previous run
        if best_path.exists():
            print("  Loading best model from disk (no improvement this session)")
            best_ckpt = torch.load(best_path, map_location=device)
            best_state = {
                "classifier": best_ckpt["classifier"],
                "rl_layer": best_ckpt["rl_layer"],
                "epoch": best_ckpt["best_epoch"],
                "threshold": best_ckpt.get("threshold", 0.5),
            }
            best_train_metrics = EvalMetrics(**best_ckpt["best_train_metrics"]) if "best_train_metrics" in best_ckpt else None
            del best_ckpt
        else:
            raise RuntimeError("Training did not produce a valid checkpoint.")

    unwrap_module(classifier).load_state_dict(best_state["classifier"])
    unwrap_module(rl_layer).load_state_dict(best_state["rl_layer"])

    _, test_probs, test_labels = collect_predictions(classifier, rl_layer, test_loader, loss_fn, device)
    test_metrics = compute_metrics(test_labels, test_probs, best_state["threshold"])

    torch.save(
        {
            "classifier": unwrap_module(classifier).state_dict(),
            "rl_layer": unwrap_module(rl_layer).state_dict(),
            "freeze_mode": freeze_mode,
            "k_value": k_value,
            "threshold": best_state["threshold"],
            "best_epoch": best_state["epoch"],
        },
        output_dir / f"{run_name}.pth",
    )

    with open(output_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    result = {
        "freeze_mode": freeze_mode,
        "k_value": k_value,
        "train_counts": summarize_counts(sampled_train),
        "test_counts": summarize_counts(test_info),
        "per_class_used": {CLASS_NAMES[k]: int(v) for k, v in per_class_used.items()},
        "best_epoch": int(best_state["epoch"]),
        "best_threshold": float(best_state["threshold"]),
        "trainable_params": int(sum(p.numel() for p in unwrap_module(classifier).parameters() if p.requires_grad) + sum(p.numel() for p in unwrap_module(rl_layer).parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in unwrap_module(classifier).parameters()) + sum(p.numel() for p in unwrap_module(rl_layer).parameters())),
        "train_metrics": asdict(best_train_metrics),
        "test_metrics": asdict(test_metrics),
    }

    print(
        f"  Test | Acc {test_metrics.accuracy:.4f} | F1 {test_metrics.f1:.4f} | "
        f"Prec {test_metrics.precision:.4f} | Sens {test_metrics.sensitivity:.4f} | "
        f"FAR {test_metrics.false_alarm_rate:.4f} ({test_metrics.false_alarms} FP)"
    )

    del classifier, rl_layer, backbone, train_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"Visible CUDA GPUs: {available_gpus}")
        if available_gpus >= 4:
            data_parallel_device_ids = [0, 1, 2, 3]
            print("Using DataParallel on GPUs: [0, 1, 2, 3]")
        else:
            data_parallel_device_ids = None
            print("Fewer than 4 GPUs visible; using single-GPU mode.")
    else:
        data_parallel_device_ids = None
    print(f"TUAB base: {BASE}")

    mean = np.load(NORM_STATS_DIR / "mean.npy")
    std = np.load(NORM_STATS_DIR / "std.npy")
    print(f"Normalization shape: mean={mean.shape}, std={std.shape}")

    train_info = []
    test_info = []
    for label, directory in TRAIN_DIRS.items():
        info = build_sample_index(directory, label)
        train_info.extend(info)
        print(f"{directory.name}: {len(info):,} samples")

    for label, directory in TEST_DIRS.items():
        info = build_sample_index(directory, label)
        test_info.extend(info)
        print(f"{directory.name}: {len(info):,} samples")

    train_pool = train_info
    print(f"Train pool (predefined train folders): {len(train_pool):,} samples")
    print(f"Test set: {len(test_info):,} samples")

    if RESUME_DIR is not None:
        output_dir = Path(RESUME_DIR)
        print(f"Resuming into existing output dir: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./tuab_fewshot_rl_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for freeze_mode in FREEZE_MODES:
        print(f"\n{'=' * 80}")
        print(f"Mode: {freeze_mode.upper()}")
        print(f"{'=' * 80}")
        for k_value in K_VALUES:
            print(f"\nRunning K={'all' if k_value <= 0 else k_value}")
            result = run_experiment(
                train_pool=train_pool,
                test_info=test_info,
                mean=mean,
                std=std,
                k_value=k_value,
                freeze_mode=freeze_mode,
                device=device,
                output_dir=output_dir,
                data_parallel_device_ids=data_parallel_device_ids,
            )
            all_results.append(result)

    with open(output_dir / "summary_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    best_result = max(all_results, key=lambda r: (r["test_metrics"]["f1"], r["test_metrics"]["balanced_accuracy"], r["test_metrics"]["roc_auc"]))

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Mode':<12} {'K':<6} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Sens':<8} {'Spec':<8} {'FAR':<8}")
    print("-" * 80)
    for result in all_results:
        metrics = result["test_metrics"]
        print(
            f"{result['freeze_mode']:<12} {result['k_value']:<6} "
            f"{metrics['accuracy']:<8.4f} {metrics['f1']:<8.4f} {metrics['precision']:<8.4f} "
            f"{metrics['sensitivity']:<8.4f} {metrics['specificity']:<8.4f} {metrics['false_alarm_rate']:<8.4f}"
        )

    print("\nBest test configuration:")
    print(
        f"  Mode={best_result['freeze_mode']} | K={best_result['k_value']} | "
        f"F1={best_result['test_metrics']['f1']:.4f} | "
        f"Acc={best_result['test_metrics']['accuracy']:.4f} | "
        f"ROC-AUC={best_result['test_metrics']['roc_auc']:.4f}"
    )
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
