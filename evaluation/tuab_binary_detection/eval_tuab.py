#!/usr/bin/env python3
"""Evaluate the trained TUAB model on test data (no training)."""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from fewshot_tuab_rl import (
    ARCH,
    BASE,
    CLASS_NAMES,
    TEST_DIRS,
    NORM_STATS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    RL_WEIGHT,
    EEGClassifier,
    ReinforcedDecisionLayer,
    MemmapEEGDataset,
    build_sample_index,
    collate_fn,
    compute_metrics,
    find_best_threshold,
    set_seed,
)
from Diff_EEG_train import DeepEnhancedEEGDiffusionModel
from torch.utils.data import DataLoader


CHECKPOINT = Path("/home/abdulh/scratch/tuab_fewshot_rl_20260408_072502/kall_unfrozen_best.pth")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    threshold = ckpt.get("threshold", 0.5)
    best_epoch = ckpt.get("best_epoch", "?")
    freeze_mode = ckpt.get("freeze_mode", "unfrozen")
    print(f"  Best epoch: {best_epoch}, threshold: {threshold}, mode: {freeze_mode}")

    # Build model
    backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
    classifier = EEGClassifier(backbone=backbone, num_classes=2, dropout=0.4).to(device)
    agg_dim = ARCH["model_channels"] * sum(ARCH["channel_multipliers"]) * 5
    rl_layer = ReinforcedDecisionLayer(input_dim=agg_dim, rl_weight=RL_WEIGHT).to(device)

    classifier.load_state_dict(ckpt["classifier"])
    rl_layer.load_state_dict(ckpt["rl_layer"])
    del ckpt
    print("Model loaded.\n")

    # Load normalization stats
    mean = np.load(NORM_STATS_DIR / "mean.npy")
    std = np.load(NORM_STATS_DIR / "std.npy")

    # Build test set
    test_info = []
    for label, directory in TEST_DIRS.items():
        info = build_sample_index(directory, label)
        test_info.extend(info)
        print(f"{directory.name}: {len(info):,} samples")
    print(f"Total test samples: {len(test_info):,}\n")

    test_dataset = MemmapEEGDataset(test_info, mean, std)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # Inference
    classifier.eval()
    rl_layer.eval()
    probs_all = []
    labels_all = []

    print("Running inference...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            logits, features = classifier(x)
            adjusted, probs, _, _ = rl_layer(logits, features, training=False)
            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(y.numpy())
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(test_loader)}")

    probs_arr = np.asarray(probs_all)
    labels_arr = np.asarray(labels_all)

    # Metrics at saved threshold
    metrics_saved = compute_metrics(labels_arr, probs_arr, threshold)
    # Also find optimal threshold
    best_thr, metrics_best = find_best_threshold(labels_arr, probs_arr)

    print(f"\n{'='*70}")
    print(f"TEST RESULTS  (saved threshold = {threshold})")
    print(f"{'='*70}")
    print(f"  Accuracy:      {metrics_saved.accuracy:.4f}")
    print(f"  F1:            {metrics_saved.f1:.4f}")
    print(f"  Precision:     {metrics_saved.precision:.4f}")
    print(f"  Sensitivity:   {metrics_saved.sensitivity:.4f}")
    print(f"  Specificity:   {metrics_saved.specificity:.4f}")
    print(f"  Balanced Acc:  {metrics_saved.balanced_accuracy:.4f}")
    print(f"  ROC-AUC:       {metrics_saved.roc_auc:.4f}")
    print(f"  PR-AUC:        {metrics_saved.pr_auc:.4f}")
    print(f"  False Alarms:  {metrics_saved.false_alarms} (FAR={metrics_saved.false_alarm_rate:.4f})")
    print(f"  TP={metrics_saved.tp}  TN={metrics_saved.tn}  FP={metrics_saved.fp}  FN={metrics_saved.fn}")

    print(f"\n{'='*70}")
    print(f"TEST RESULTS  (optimal threshold = {best_thr:.2f})")
    print(f"{'='*70}")
    print(f"  Accuracy:      {metrics_best.accuracy:.4f}")
    print(f"  F1:            {metrics_best.f1:.4f}")
    print(f"  Precision:     {metrics_best.precision:.4f}")
    print(f"  Sensitivity:   {metrics_best.sensitivity:.4f}")
    print(f"  Specificity:   {metrics_best.specificity:.4f}")
    print(f"  Balanced Acc:  {metrics_best.balanced_accuracy:.4f}")
    print(f"  ROC-AUC:       {metrics_best.roc_auc:.4f}")
    print(f"  PR-AUC:        {metrics_best.pr_auc:.4f}")
    print(f"  False Alarms:  {metrics_best.false_alarms} (FAR={metrics_best.false_alarm_rate:.4f})")
    print(f"  TP={metrics_best.tp}  TN={metrics_best.tn}  FP={metrics_best.fp}  FN={metrics_best.fn}")

    # Save results
    out_path = CHECKPOINT.parent / "test_results.json"
    results = {
        "checkpoint": str(CHECKPOINT),
        "best_epoch": best_epoch,
        "num_test_samples": len(labels_arr),
        "saved_threshold": asdict(metrics_saved),
        "optimal_threshold": asdict(metrics_best),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
