

#!/usr/bin/env python3
"""
Segment-Wise Few-Shot Seizure Subtype Classification
=====================================================
Few-shot multi-class seizure subtype classification using DiffEEG backbone.
- Compare: Frozen vs Unfrozen backbone
- Test multiple K values (capped at half of smallest class)
- Delete last 3 subtypes (lowest samples) → keep 5 classes

Dataset: THUSZ (segment-wise stratified split)
Task: 5-class seizure subtype classification
Best result: Unfrozen, K=342, F1=0.6578, Accuracy=0.6564

No saved model (few-shot experiment). See results.out for full output.
- Stratified split with data augmentation
- Track F1 ≥ 0.60 as minimum target (not early-stop)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from Diff_EEG_train import DeepEnhancedEEGDiffusionModel

# ================================================================
# CONFIGURATION
# ================================================================
BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/seizure_subtypes"
DIFFUSION_CHECKPOINT = "/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth"
NORM_STATS_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization"

ARCH = dict(
    in_channels=22,
    model_channels=32,
    channel_multipliers=[1, 2, 4, 8],
    num_res_blocks=2,
    time_emb_dim=512,
    dropout=0.1,
    attention_heads=8,
)

BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 15

# Target F1-score (minimum acceptable, NOT a stopping condition)
TARGET_F1_MIN = 0.60

# Base K values to test (will be filtered by data availability)
BASE_K_VALUES = [10, 50, 100, 200, 400, 500, 800]
FREEZE_MODES = ['frozen', 'unfrozen']

# ================================================================
# DATA LOADING
# ================================================================
print("Loading normalization stats...")
mean = np.load(Path(NORM_STATS_DIR) / "mean.npy")
std = np.load(Path(NORM_STATS_DIR) / "std.npy")
print("✓ Loaded\n")

def normalize(data):
    """Normalize batch (B, C, T)."""
    if data.ndim == 2:
        m = mean.reshape(-1, 1)
        s = std.reshape(-1, 1)
    else:
        m = mean.reshape(1, -1, 1)
        s = std.reshape(1, -1, 1)
    return ((data - m) / (s + 1e-6)).astype(np.float32)

def augment_data(data, noise_level=0.01):
    """Simple augmentation: add small gaussian noise."""
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return (data + noise).astype(np.float32)

def load_all_subtypes():
    """Load all subtypes, delete last 3 (lowest samples)."""
    print("="*70)
    print("LOADING ALL SUBTYPES")
    print("="*70 + "\n")
    
    all_data = {}
    subtype_counts = {}
    
    base_path = Path(BASE)
    
    # Dynamically find all subtypes
    subtype_dirs = sorted([d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith('subtype')])
    
    num_subtypes = len(subtype_dirs)
    print(f"Found {num_subtypes} subtypes\n")
    
    for subtype_path in subtype_dirs:
        subtype_name = subtype_path.name
        data_file = subtype_path / "data.npy"
        
        if not data_file.exists():
            print(f"Warning: data.npy not found in {subtype_name}")
            continue
        
        # Load and normalize
        data = np.load(data_file, allow_pickle=False)
        data = normalize(data)
        
        subtype_id = int(subtype_name.split('_')[1]) - 1  # 0-indexed
        all_data[subtype_id] = data
        subtype_counts[subtype_id] = len(data)
        
        print(f"  {subtype_name}: {len(data):,} samples")
    
    # Analyze distribution
    print(f"\nSubtype distribution (before deletion):")
    sorted_by_count = sorted(subtype_counts.items(), key=lambda x: x[1])
    for subtype_id, count in sorted_by_count:
        print(f"  Subtype {subtype_id+1:02d}: {count:,} samples")
    
    # Delete last 3 subtypes with least samples
    subtypes_to_delete = [s[0] for s in sorted_by_count[:3]]  # Last 3 (least samples)
    print(f"\nDeleting subtypes {sorted([s+1 for s in subtypes_to_delete])} (lowest sample counts)")
    
    for subtype_id in subtypes_to_delete:
        if subtype_id in all_data:
            del all_data[subtype_id]
    
    return all_data, num_subtypes

# ================================================================
# MODELS
# ================================================================
class EEGClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=256, num_classes=5, dropout=0.3):
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


def train_epoch(classifier, loader, optimizer, loss_fn, device):
    classifier.train()
    
    total_loss, preds_all, labels_all = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits, _ = classifier(x)
        loss = loss_fn(logits, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()
        
        total_loss.append(loss.item())
        with torch.no_grad():
            preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    
    acc = accuracy_score(labels_all, preds_all)
    return float(np.mean(total_loss)), acc


@torch.no_grad()
def evaluate(classifier, loader, device):
    classifier.eval()
    preds_all, labels_all = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = classifier(x)
        preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    
    return np.array(preds_all), np.array(labels_all)


def setup_backbone(backbone, freeze_mode):
    """Setup backbone freezing/unfreezing."""
    if freeze_mode == 'frozen':
        for p in backbone.parameters():
            p.requires_grad = False
    else:  # unfrozen
        # Unfreeze last 2 encoder blocks
        for p in backbone.parameters():
            p.requires_grad = False
        
        n_blocks = len(backbone.down_blocks)
        blocks_to_unfreeze = min(2, n_blocks)
        for bl in backbone.down_blocks[-blocks_to_unfreeze:]:
            for block in bl:
                for p in block.parameters():
                    p.requires_grad = True
        
        # Also unfreeze time embedding
        for p in backbone.time_mlp.parameters():
            p.requires_grad = True


# ================================================================
# MAIN
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load all subtypes (with deletion of last 3)
all_data, num_subtypes = load_all_subtypes()

# Create full dataset with class labels (0-indexed)
X_all = []
y_all = []

# Map old subtype IDs to new class IDs (0, 1, 2, 3, 4)
new_class_id = 0
for subtype_id in sorted(all_data.keys()):
    data = all_data[subtype_id]
    X_all.append(data)
    y_all.extend([new_class_id] * len(data))
    new_class_id += 1

X_all = np.concatenate(X_all, axis=0)
y_all = np.array(y_all, dtype=np.int64)

num_classes = len(np.unique(y_all))

print(f"\n✓ After deletion: {num_classes} classes remaining\n")
print(f"Total samples: {len(X_all):,}")
print(f"Final number of classes: {num_classes}")

for class_id in range(num_classes):
    count = np.sum(y_all == class_id)
    print(f"  Class {class_id}: {count:,} samples")

# Stratified train/test split
print(f"\n{'='*70}")
print("CREATING STRATIFIED TRAIN/TEST SPLIT (80/20)")
print(f"{'='*70}\n")

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Train pool: {len(X_train_all):,} samples")
print(f"Test set: {len(X_test):,} samples")

for class_id in range(num_classes):
    train_count = np.sum(y_train_all == class_id)
    test_count = np.sum(y_test == class_id)
    print(f"  Class {class_id}: {train_count:,} train, {test_count:,} test")

# 🎯 Compute feasible K values based on smallest class (half buffer)
print(f"\n{'='*70}")
print("COMPUTING FEASIBLE K VALUES")
print(f"{'='*70}\n")

class_sizes_train = [np.sum(y_train_all == c) for c in range(num_classes)]
MIN_CLASS_SIZE = min(class_sizes_train)
MAX_K_PER_CLASS = MIN_CLASS_SIZE // 2  # Half of smallest class for training buffer

# Always include K=250 if possible, and always include MAX_K_PER_CLASS
K_VALUES = [k for k in BASE_K_VALUES if k <= MAX_K_PER_CLASS]
if 250 <= MAX_K_PER_CLASS and 250 not in K_VALUES:
    K_VALUES.append(250)
if MAX_K_PER_CLASS not in K_VALUES:
    K_VALUES.append(MAX_K_PER_CLASS)
K_VALUES = sorted(set(K_VALUES))
if not K_VALUES:
    K_VALUES = [min(BASE_K_VALUES)]  # Fallback to smallest if none fit
    print(f"⚠ Warning: No base K values fit within data constraints. Using minimum: {K_VALUES[0]}")

print(f"📊 Train set class sizes: {[f'{s:,}' for s in class_sizes_train]}")
print(f"📊 Smallest class size: {MIN_CLASS_SIZE:,} → Max K (half): {MAX_K_PER_CLASS}")
print(f"🎯 Testing K values: {K_VALUES}")
print(f"🎯 F1-Score target (minimum): ≥ {TARGET_F1_MIN}\n")

# Load backbone once
print(f"{'='*70}")
print("LOADING BACKBONE")
print(f"{'='*70}\n")
backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
ckpt = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
backbone.load_state_dict(ckpt['model_state_dict'])
print("✓ Loaded\n")

# Create output directory
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base = Path(f"./seizure_subtype_comparison_{ts}")
output_base.mkdir(exist_ok=True)

# Store all results
all_results = []

# ================================================================
# EXPERIMENTS
# ================================================================
print("="*70)
print("RUNNING EXPERIMENTS: FROZEN vs UNFROZEN × K VALUES")
print("="*70 + "\n")

for freeze_mode in FREEZE_MODES:
    print(f"\n{'='*70}")
    print(f"MODE: {freeze_mode.upper()}")
    print(f"{'='*70}\n")
    
    for k_value in K_VALUES:
        print(f"\n--- K={k_value} ---")
        
        # For each class, sample up to k_value or as many as available, and augment if class is small
        X_train_list, y_train_list = [], []
        rng_seed = 42 + k_value + (0 if freeze_mode == 'frozen' else 1000)
        rng = np.random.default_rng(rng_seed)
        for class_id in range(num_classes):
            class_mask = y_train_all == class_id
            class_indices = np.where(class_mask)[0]
            available = len(class_indices)
            k_this = min(k_value, available)
            if k_this < k_value:
                print(f"  ⚠ Class {class_id}: requested K={k_value}, available={available}, using {k_this}")
            if k_this > 0:
                sampled_indices = rng.choice(class_indices, size=k_this, replace=False)
                sampled_data = X_train_all[sampled_indices]
                X_train_list.append(sampled_data)
                y_train_list.extend([class_id] * k_this)
                # Augment if class is small (less than 2000)
                if available < 2000 and k_this > 10:
                    n_augment = min(k_this // 3, 20)
                    if n_augment > 0:
                        augment_indices = rng.choice(sampled_indices, size=n_augment, replace=True)
                        augmented_data = np.array([augment_data(X_train_all[i]) for i in augment_indices])
                        X_train_list.append(augmented_data)
                        y_train_list.extend([class_id] * n_augment)
                        print(f"    → Added {n_augment} augmented samples for Class {class_id}")
        if not X_train_list:
            print(f"  ❌ No training data for K={k_value}. Skipping.")
            continue
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.array(y_train_list, dtype=np.int64)
        print(f"  Train samples: {len(X_train):,} (K={k_value} base + augmented)")
        
        # Data loaders
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                                 batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Create classifier
        classifier = EEGClassifier(backbone, feature_dim=256, num_classes=num_classes, dropout=0.3).to(device)
        setup_backbone(classifier.backbone, freeze_mode)

        # === Load weights from seizure vs non-seizure classifier and replace last layer ===
        PRETRAINED_CLF_PATH = "/home/abdulh/scratch/seizure_clf_20260226_083908/best_classifier.pth"
        try:
            pre_ckpt = torch.load(PRETRAINED_CLF_PATH, map_location=device)
            # Load backbone weights (if not already loaded)
            if hasattr(classifier, 'backbone') and hasattr(pre_ckpt, 'get'):
                if 'backbone' in pre_ckpt:
                    classifier.backbone.load_state_dict(pre_ckpt['backbone'], strict=False)
            # Load classifier weights except last layer
            pretrained_dict = pre_ckpt['classifier'] if 'classifier' in pre_ckpt else pre_ckpt
            model_dict = classifier.classifier.state_dict()
            # Remove last layer weights (for 2-class)
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and '4.weight' not in k and '4.bias' not in k}
            model_dict.update(filtered_dict)
            classifier.classifier.load_state_dict(model_dict, strict=False)
            print(f"✓ Loaded pretrained classifier weights (except last layer) from {PRETRAINED_CLF_PATH}")
        except Exception as e:
            print(f"⚠ Could not load pretrained classifier weights: {e}")

        # Replace last layer for 5-class output (find last nn.Linear dynamically)
        with torch.no_grad():
            last_linear_idx = None
            last_linear_out = None
            for idx in reversed(range(len(classifier.classifier))):
                if isinstance(classifier.classifier[idx], nn.Linear):
                    last_linear_idx = idx
                    last_linear_out = classifier.classifier[idx].in_features
                    break
            if last_linear_idx is not None:
                classifier.classifier[last_linear_idx] = nn.Linear(last_linear_out, num_classes).to(device)
                print(f"✓ Replaced last classifier layer for {num_classes}-class output at index {last_linear_idx}")
            else:
                print(f"⚠ Could not find last nn.Linear layer in classifier to replace.")
        
        # Count trainable params
        trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in classifier.parameters())
        
        # Setup optimizer
        if freeze_mode == 'frozen':
            optimizer = optim.AdamW([{'params': classifier.classifier.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}])
        else:  # unfrozen
            backbone_params = [p for n, p in classifier.named_parameters() 
                             if p.requires_grad and 'backbone' in n]
            head_params = [p for n, p in classifier.named_parameters() 
                         if p.requires_grad and 'backbone' not in n]
            
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
                {'params': head_params, 'lr': 5e-4, 'weight_decay': 1e-4},
            ])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum() * num_classes
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop with early stopping
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_test_preds, best_test_labels = None, None
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            tr_loss, tr_acc = train_epoch(classifier, train_loader, optimizer, loss_fn, device)
            scheduler.step()
            
            test_preds, test_labels = evaluate(classifier, test_loader, device)
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_test_preds = test_preds.copy()
                best_test_labels = test_labels.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  → Early stop at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                    break
        
        # Store result
        result = {
            'freeze_mode': freeze_mode,
            'k_value': k_value,
            'test_accuracy': float(best_test_acc),
            'test_f1': float(best_test_f1),
            'meets_target': bool(best_test_f1 >= TARGET_F1_MIN),
            'trainable_params': int(trainable_params),
            'total_params': int(total_params),
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'epochs_trained': epoch + 1,
            'class_sizes_used': {c: int(min(k_value, class_sizes_train[c]//2)) for c in range(num_classes)},
        }
        
        all_results.append(result)
        
        # Print result with target status
        status_icon = "✅" if best_test_f1 >= TARGET_F1_MIN else "❌"
        print(f"  {status_icon} Accuracy: {best_test_acc:.4f}, F1: {best_test_f1:.4f} "
              f"(target ≥ {TARGET_F1_MIN})")
        print(f"      Params: {trainable_params:,}/{total_params:,} trainable/total")
        
        # Optional: Print per-class F1 for debugging smallest class
        if k_value >= 100:  # Only for larger K to reduce output
            report = classification_report(best_test_labels, best_test_preds, 
                                         target_names=[f"C{i}" for i in range(num_classes)],
                                         output_dict=True, zero_division=0)
            class_f1s = [report[f"C{i}"]['f1-score'] for i in range(num_classes)]
            print(f"      Per-class F1: {[f'{f:.3f}' for f in class_f1s]}")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n\n{'='*70}")
print(f"SUMMARY: F1-Score Target ≥ {TARGET_F1_MIN}")
print(f"{'='*70}\n")

print(f"{'Mode':<12} {'K':<6} {'Accuracy':<12} {'F1-Score':<12} {'Meets Target':<15} {'Epochs':<8}")
print("-" * 75)

for result in sorted(all_results, key=lambda x: (x['freeze_mode'], x['k_value'])):
    status = "✅ YES" if result['meets_target'] else "❌ no"
    print(f"{result['freeze_mode']:<12} {result['k_value']:<6} "
          f"{result['test_accuracy']:<12.4f} {result['test_f1']:<12.4f} "
          f"{status:<15} {result['epochs_trained']:<8}")

# Find best result that meets target (if any)
valid_results = [r for r in all_results if r['meets_target']]
if valid_results:
    best_valid = max(valid_results, key=lambda x: x['test_f1'])
    print(f"\n🏆 Best Result Meeting Target (F1 ≥ {TARGET_F1_MIN}):")
    print(f"   Mode: {best_valid['freeze_mode']} | K: {best_valid['k_value']}")
    print(f"   Accuracy: {best_valid['test_accuracy']:.4f} | F1: {best_valid['test_f1']:.4f}")
    print(f"   Trainable params: {best_valid['trainable_params']:,}")
else:
    print(f"\n⚠ No configuration reached F1 ≥ {TARGET_F1_MIN}.")
    print("   Suggestions:")
    print("   • Try unfrozen mode if you only tested frozen")
    print("   • Increase augmentation for smallest classes")
    print("   • Train longer (increase NUM_EPOCHS or PATIENCE)")
    print("   • Consider feature extraction + simpler classifier")

# Find overall best (regardless of target)
best_overall = max(all_results, key=lambda x: x['test_f1'])
print(f"\n📈 Overall Best F1 (any config):")
print(f"   Mode: {best_overall['freeze_mode']} | K: {best_overall['k_value']}")
print(f"   F1: {best_overall['test_f1']:.4f} | Accuracy: {best_overall['test_accuracy']:.4f}")

# Save results
with open(output_base / "comparison_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

frozen_results = [r for r in all_results if r['freeze_mode'] == 'frozen']
unfrozen_results = [r for r in all_results if r['freeze_mode'] == 'unfrozen']

def extract_metrics(results, ks):
    accs, f1s = [], []
    for k in ks:
        match = next((r for r in results if r['k_value'] == k), None)
        if match:
            accs.append(match['test_accuracy'])
            f1s.append(match['test_f1'])
        else:
            accs.append(None)
            f1s.append(None)
    return accs, f1s

# Plot Accuracy
frozen_ks = sorted(set(r['k_value'] for r in frozen_results))
frozen_accs, frozen_f1s = extract_metrics(frozen_results, frozen_ks)
unfrozen_ks = sorted(set(r['k_value'] for r in unfrozen_results))
unfrozen_accs, unfrozen_f1s = extract_metrics(unfrozen_results, unfrozen_ks)

axes[0].plot(frozen_ks, frozen_accs, 'o-', linewidth=2, markersize=8, label='Frozen Backbone')
axes[0].plot(unfrozen_ks, unfrozen_accs, 's-', linewidth=2, markersize=8, label='Unfrozen Backbone')
axes[0].axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='F1 Target (ref)')
axes[0].set_xlabel('K (Samples per Class)', fontsize=12)
axes[0].set_ylabel('Test Accuracy', fontsize=12)
axes[0].set_title('Accuracy vs K Value', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(K_VALUES)

# Plot F1-Score with target line
axes[1].plot(frozen_ks, frozen_f1s, 'o-', linewidth=2, markersize=8, label='Frozen Backbone')
axes[1].plot(unfrozen_ks, unfrozen_f1s, 's-', linewidth=2, markersize=8, label='Unfrozen Backbone')
axes[1].axhline(y=TARGET_F1_MIN, color='green', linestyle='--', linewidth=1.5, 
                label=f'Target F1 ≥ {TARGET_F1_MIN}')
axes[1].set_xlabel('K (Samples per Class)', fontsize=12)
axes[1].set_ylabel('F1-Score (Weighted)', fontsize=12)
axes[1].set_title('F1-Score vs K Value', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
axes[1].set_xticks(K_VALUES)

plt.tight_layout()
plt.savefig(output_base / "comparison_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# Save detailed per-class report for best run
if best_test_preds is not None:
    report = classification_report(best_test_labels, best_test_preds, 
                                 target_names=[f"Class_{i}" for i in range(num_classes)],
                                 output_dict=True, zero_division=0)
    with open(output_base / "classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)

print(f"\n✓ Results saved to {output_base}/")
print(f"  - comparison_results.json")
print(f"  - comparison_plots.png")
print(f"  - classification_report.json (per-class metrics)")
print(f"\n🎯 Remember: F1 ≥ {TARGET_F1_MIN} is your minimum target, not a stopping condition.")



# ********************************************************************


# #!/usr/bin/env python3
# """
# Few-Shot Multi-Class Seizure Subtype Classification
# - Compare: Frozen vs Unfrozen backbone
# - Test multiple K values (capped at half of smallest class)
# - Delete last 3 subtypes (lowest samples) → keep 5 classes
# - Stratified split with data augmentation
# - Track F1 ≥ 0.60 as minimum target (not early-stop)
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import json
# from datetime import datetime
# from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score)
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')

# from Diff_EEG_train import DeepEnhancedEEGDiffusionModel

# # ================================================================
# # CONFIGURATION
# # ================================================================
# BASE = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/seizure_subtypes"
# DIFFUSION_CHECKPOINT = "/home/abdulh/scratch/training_diffusion2/best_EEGDIFF2.pth"
# NORM_STATS_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization"

# ARCH = dict(in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
#             num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8)

# BATCH_SIZE = 32
# NUM_EPOCHS = 100
# PATIENCE = 15

# # Target F1-score (minimum acceptable, NOT a stopping condition)
# TARGET_F1_MIN = 0.60

# # Base K values to test (will be filtered by data availability)
# BASE_K_VALUES = [10, 50, 100, 200, 400, 500, 800]
# FREEZE_MODES = ['frozen', 'unfrozen']

# # ================================================================
# # DATA LOADING
# # ================================================================
# print("Loading normalization stats...")
# mean = np.load(Path(NORM_STATS_DIR) / "mean.npy")
# std = np.load(Path(NORM_STATS_DIR) / "std.npy")
# print("✓ Loaded\n")

# def normalize(data):
#     """Normalize batch (B, C, T)."""
#     if data.ndim == 2:
#         m = mean.reshape(-1, 1)
#         s = std.reshape(-1, 1)
#     else:
#         m = mean.reshape(1, -1, 1)
#         s = std.reshape(1, -1, 1)
#     return ((data - m) / (s + 1e-6)).astype(np.float32)

# def augment_data(data, noise_level=0.01):
#     """Simple augmentation: add small gaussian noise."""
#     noise = np.random.normal(0, noise_level * np.std(data), data.shape)
#     return (data + noise).astype(np.float32)

# def load_all_subtypes():
#     """Load all subtypes, delete last 3 (lowest samples)."""
#     print("="*70)
#     print("LOADING ALL SUBTYPES")
#     print("="*70 + "\n")
    
#     all_data = {}
#     subtype_counts = {}
    
#     base_path = Path(BASE)
    
#     # Dynamically find all subtypes
#     subtype_dirs = sorted([d for d in base_path.iterdir() 
#                           if d.is_dir() and d.name.startswith('subtype')])
    
#     num_subtypes = len(subtype_dirs)
#     print(f"Found {num_subtypes} subtypes\n")
    
#     for subtype_path in subtype_dirs:
#         subtype_name = subtype_path.name
#         data_file = subtype_path / "data.npy"
        
#         if not data_file.exists():
#             print(f"Warning: data.npy not found in {subtype_name}")
#             continue
        
#         # Load and normalize
#         data = np.load(data_file, allow_pickle=False)
#         data = normalize(data)
        
#         subtype_id = int(subtype_name.split('_')[1]) - 1  # 0-indexed
#         all_data[subtype_id] = data
#         subtype_counts[subtype_id] = len(data)
        
#         print(f"  {subtype_name}: {len(data):,} samples")
    
#     # Analyze distribution
#     print(f"\nSubtype distribution (before deletion):")
#     sorted_by_count = sorted(subtype_counts.items(), key=lambda x: x[1])
#     for subtype_id, count in sorted_by_count:
#         print(f"  Subtype {subtype_id+1:02d}: {count:,} samples")
    
#     # Delete last 3 subtypes with least samples
#     subtypes_to_delete = [s[0] for s in sorted_by_count[:3]]  # Last 3 (least samples)
#     print(f"\nDeleting subtypes {sorted([s+1 for s in subtypes_to_delete])} (lowest sample counts)")
    
#     for subtype_id in subtypes_to_delete:
#         if subtype_id in all_data:
#             del all_data[subtype_id]
    
#     return all_data, num_subtypes

# # ================================================================
# # MODELS
# # ================================================================
# class EEGClassifier(nn.Module):
#     def __init__(self, backbone, feature_dim=256, num_classes=5, dropout=0.3):
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


# def train_epoch(classifier, loader, optimizer, loss_fn, device):
#     classifier.train()
    
#     total_loss, preds_all, labels_all = [], [], []
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
        
#         logits, _ = classifier(x)
#         loss = loss_fn(logits, y)
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
#         optimizer.step()
        
#         total_loss.append(loss.item())
#         with torch.no_grad():
#             preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
#             labels_all.extend(y.cpu().numpy())
    
#     acc = accuracy_score(labels_all, preds_all)
#     return float(np.mean(total_loss)), acc


# @torch.no_grad()
# def evaluate(classifier, loader, device):
#     classifier.eval()
#     preds_all, labels_all = [], []
    
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         logits, _ = classifier(x)
#         preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
#         labels_all.extend(y.cpu().numpy())
    
#     return np.array(preds_all), np.array(labels_all)


# def setup_backbone(backbone, freeze_mode):
#     """Setup backbone freezing/unfreezing."""
#     if freeze_mode == 'frozen':
#         for p in backbone.parameters():
#             p.requires_grad = False
#     else:  # unfrozen
#         # Unfreeze last 2 encoder blocks
#         for p in backbone.parameters():
#             p.requires_grad = False
        
#         n_blocks = len(backbone.down_blocks)
#         blocks_to_unfreeze = min(2, n_blocks)
#         for bl in backbone.down_blocks[-blocks_to_unfreeze:]:
#             for block in bl:
#                 for p in block.parameters():
#                     p.requires_grad = True
        
#         # Also unfreeze time embedding
#         for p in backbone.time_mlp.parameters():
#             p.requires_grad = True


# # ================================================================
# # MAIN
# # ================================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}\n")

# # Load all subtypes (with deletion of last 3)
# all_data, num_subtypes = load_all_subtypes()

# # Create full dataset with class labels (0-indexed)
# X_all = []
# y_all = []

# # Map old subtype IDs to new class IDs (0, 1, 2, 3, 4)
# new_class_id = 0
# for subtype_id in sorted(all_data.keys()):
#     data = all_data[subtype_id]
#     X_all.append(data)
#     y_all.extend([new_class_id] * len(data))
#     new_class_id += 1

# X_all = np.concatenate(X_all, axis=0)
# y_all = np.array(y_all, dtype=np.int64)

# num_classes = len(np.unique(y_all))

# print(f"\n✓ After deletion: {num_classes} classes remaining\n")
# print(f"Total samples: {len(X_all):,}")
# print(f"Final number of classes: {num_classes}")

# for class_id in range(num_classes):
#     count = np.sum(y_all == class_id)
#     print(f"  Class {class_id}: {count:,} samples")

# # Stratified train/test split
# print(f"\n{'='*70}")
# print("CREATING STRATIFIED TRAIN/TEST SPLIT (80/20)")
# print(f"{'='*70}\n")

# X_train_all, X_test, y_train_all, y_test = train_test_split(
#     X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
# )

# print(f"Train pool: {len(X_train_all):,} samples")
# print(f"Test set: {len(X_test):,} samples")

# for class_id in range(num_classes):
#     train_count = np.sum(y_train_all == class_id)
#     test_count = np.sum(y_test == class_id)
#     print(f"  Class {class_id}: {train_count:,} train, {test_count:,} test")

# # 🎯 Compute feasible K values based on smallest class (half buffer)
# print(f"\n{'='*70}")
# print("COMPUTING FEASIBLE K VALUES")
# print(f"{'='*70}\n")

# class_sizes_train = [np.sum(y_train_all == c) for c in range(num_classes)]
# MIN_CLASS_SIZE = min(class_sizes_train)
# MAX_K_PER_CLASS = MIN_CLASS_SIZE // 2  # Half of smallest class for training buffer

# # Filter K values to only those feasible given data constraints
# K_VALUES = [k for k in BASE_K_VALUES if k <= MAX_K_PER_CLASS]
# if not K_VALUES:
#     K_VALUES = [min(BASE_K_VALUES)]  # Fallback to smallest if none fit
#     print(f"⚠ Warning: No base K values fit within data constraints. Using minimum: {K_VALUES[0]}")

# print(f"📊 Train set class sizes: {[f'{s:,}' for s in class_sizes_train]}")
# print(f"📊 Smallest class size: {MIN_CLASS_SIZE:,} → Max K (half): {MAX_K_PER_CLASS}")
# print(f"🎯 Testing K values: {K_VALUES}")
# print(f"🎯 F1-Score target (minimum): ≥ {TARGET_F1_MIN}\n")

# # Load backbone once
# print(f"{'='*70}")
# print("LOADING BACKBONE")
# print(f"{'='*70}\n")
# backbone = DeepEnhancedEEGDiffusionModel(**ARCH).to(device)
# ckpt = torch.load(DIFFUSION_CHECKPOINT, map_location=device)
# backbone.load_state_dict(ckpt['model_state_dict'])
# print("✓ Loaded\n")

# # Create output directory
# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_base = Path(f"./seizure_subtype_comparison_{ts}")
# output_base.mkdir(exist_ok=True)

# # Store all results
# all_results = []

# # ================================================================
# # EXPERIMENTS
# # ================================================================
# print("="*70)
# print("RUNNING EXPERIMENTS: FROZEN vs UNFROZEN × K VALUES")
# print("="*70 + "\n")

# for freeze_mode in FREEZE_MODES:
#     print(f"\n{'='*70}")
#     print(f"MODE: {freeze_mode.upper()}")
#     print(f"{'='*70}\n")
    
#     for k_value in K_VALUES:
#         print(f"\n--- K={k_value} ---")
        
#         # Sample K per class with augmentation (capped at half of available)
#         X_train_list, y_train_list = [], []
#         # Use deterministic seed per (k_value, freeze_mode) combo
#         rng_seed = 42 + k_value + (0 if freeze_mode == 'frozen' else 1000)
#         rng = np.random.default_rng(rng_seed)
        
#         for class_id in range(num_classes):
#             class_mask = y_train_all == class_id
#             class_indices = np.where(class_mask)[0]
            
#             # Hard cap: never sample more than half of available samples per class
#             available = len(class_indices)
#             max_allowed = available // 2
#             k_use = min(k_value, max_allowed, available)
            
#             if k_use < k_value:
#                 print(f"  ⚠ Class {class_id}: requested K={k_value}, "
#                       f"available={available}, capped at {k_use} (half={max_allowed})")
            
#             # Sample without replacement
#             if k_use > 0:
#                 sampled_indices = rng.choice(class_indices, size=k_use, replace=False)
#                 sampled_data = X_train_all[sampled_indices]
                
#                 X_train_list.append(sampled_data)
#                 y_train_list.extend([class_id] * k_use)
                
#                 # Augmentation for minority classes (more aggressive for very small classes)
#                 if available < 2000 and k_use > 10:
#                     n_augment = min(k_use // 3, 20)  # Conservative augmentation ratio
#                     augment_indices = rng.choice(sampled_indices, size=n_augment, replace=True)
#                     augmented_data = np.array([augment_data(X_train_all[i]) for i in augment_indices])
#                     X_train_list.append(augmented_data)
#                     y_train_list.extend([class_id] * n_augment)
#                     print(f"    → Added {n_augment} augmented samples for Class {class_id}")
        
#         if not X_train_list:
#             print(f"  ❌ No training data for K={k_value}. Skipping.")
#             continue
            
#         X_train = np.concatenate(X_train_list, axis=0)
#         y_train = np.array(y_train_list, dtype=np.int64)
        
#         print(f"  Train samples: {len(X_train):,} ({k_value} base + augmented)")
        
#         # Data loaders
#         train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
#                                   batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#         test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
#                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
#         # Create classifier
#         classifier = EEGClassifier(backbone, feature_dim=256, num_classes=num_classes, dropout=0.3).to(device)
#         setup_backbone(classifier.backbone, freeze_mode)
        
#         # Count trainable params
#         trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
#         total_params = sum(p.numel() for p in classifier.parameters())
        
#         # Setup optimizer
#         if freeze_mode == 'frozen':
#             optimizer = optim.AdamW([{'params': classifier.classifier.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}])
#         else:  # unfrozen
#             backbone_params = [p for n, p in classifier.named_parameters() 
#                              if p.requires_grad and 'backbone' in n]
#             head_params = [p for n, p in classifier.named_parameters() 
#                          if p.requires_grad and 'backbone' not in n]
            
#             optimizer = optim.AdamW([
#                 {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
#                 {'params': head_params, 'lr': 5e-4, 'weight_decay': 1e-4},
#             ])
        
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        
#         # Class weights for imbalanced data
#         class_counts = np.bincount(y_train, minlength=num_classes)
#         class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
#         class_weights = class_weights / class_weights.sum() * num_classes
#         loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
#         # Training loop with early stopping
#         best_test_acc = 0.0
#         best_test_f1 = 0.0
#         best_test_preds, best_test_labels = None, None
#         patience_counter = 0
        
#         for epoch in range(NUM_EPOCHS):
#             tr_loss, tr_acc = train_epoch(classifier, train_loader, optimizer, loss_fn, device)
#             scheduler.step()
            
#             test_preds, test_labels = evaluate(classifier, test_loader, device)
#             test_acc = accuracy_score(test_labels, test_preds)
#             test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
            
#             if test_acc > best_test_acc:
#                 best_test_acc = test_acc
#                 best_test_f1 = test_f1
#                 best_test_preds = test_preds.copy()
#                 best_test_labels = test_labels.copy()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= PATIENCE:
#                     print(f"  → Early stop at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
#                     break
        
#         # Store result
#         result = {
#             'freeze_mode': freeze_mode,
#             'k_value': k_value,
#             'test_accuracy': float(best_test_acc),
#             'test_f1': float(best_test_f1),
#             'meets_target': bool(best_test_f1 >= TARGET_F1_MIN),
#             'trainable_params': int(trainable_params),
#             'total_params': int(total_params),
#             'train_size': int(len(X_train)),
#             'test_size': int(len(X_test)),
#             'epochs_trained': epoch + 1,
#             'class_sizes_used': {c: int(min(k_value, class_sizes_train[c]//2)) for c in range(num_classes)},
#         }
        
#         all_results.append(result)
        
#         # Print result with target status
#         status_icon = "✅" if best_test_f1 >= TARGET_F1_MIN else "❌"
#         print(f"  {status_icon} Accuracy: {best_test_acc:.4f}, F1: {best_test_f1:.4f} "
#               f"(target ≥ {TARGET_F1_MIN})")
#         print(f"      Params: {trainable_params:,}/{total_params:,} trainable/total")
        
#         # Optional: Print per-class F1 for debugging smallest class
#         if k_value >= 100:  # Only for larger K to reduce output
#             report = classification_report(best_test_labels, best_test_preds, 
#                                          target_names=[f"C{i}" for i in range(num_classes)],
#                                          output_dict=True, zero_division=0)
#             class_f1s = [report[f"C{i}"]['f1-score'] for i in range(num_classes)]
#             print(f"      Per-class F1: {[f'{f:.3f}' for f in class_f1s]}")

# # ================================================================
# # SUMMARY
# # ================================================================
# print(f"\n\n{'='*70}")
# print(f"SUMMARY: F1-Score Target ≥ {TARGET_F1_MIN}")
# print(f"{'='*70}\n")

# print(f"{'Mode':<12} {'K':<6} {'Accuracy':<12} {'F1-Score':<12} {'Meets Target':<15} {'Epochs':<8}")
# print("-" * 75)

# for result in sorted(all_results, key=lambda x: (x['freeze_mode'], x['k_value'])):
#     status = "✅ YES" if result['meets_target'] else "❌ no"
#     print(f"{result['freeze_mode']:<12} {result['k_value']:<6} "
#           f"{result['test_accuracy']:<12.4f} {result['test_f1']:<12.4f} "
#           f"{status:<15} {result['epochs_trained']:<8}")

# # Find best result that meets target (if any)
# valid_results = [r for r in all_results if r['meets_target']]
# if valid_results:
#     best_valid = max(valid_results, key=lambda x: x['test_f1'])
#     print(f"\n🏆 Best Result Meeting Target (F1 ≥ {TARGET_F1_MIN}):")
#     print(f"   Mode: {best_valid['freeze_mode']} | K: {best_valid['k_value']}")
#     print(f"   Accuracy: {best_valid['test_accuracy']:.4f} | F1: {best_valid['test_f1']:.4f}")
#     print(f"   Trainable params: {best_valid['trainable_params']:,}")
# else:
#     print(f"\n⚠ No configuration reached F1 ≥ {TARGET_F1_MIN}.")
#     print("   Suggestions:")
#     print("   • Try unfrozen mode if you only tested frozen")
#     print("   • Increase augmentation for smallest classes")
#     print("   • Train longer (increase NUM_EPOCHS or PATIENCE)")
#     print("   • Consider feature extraction + simpler classifier")

# # Find overall best (regardless of target)
# best_overall = max(all_results, key=lambda x: x['test_f1'])
# print(f"\n📈 Overall Best F1 (any config):")
# print(f"   Mode: {best_overall['freeze_mode']} | K: {best_overall['k_value']}")
# print(f"   F1: {best_overall['test_f1']:.4f} | Accuracy: {best_overall['test_accuracy']:.4f}")

# # Save results
# with open(output_base / "comparison_results.json", 'w') as f:
#     json.dump(all_results, f, indent=2)

# # Create plots
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# frozen_results = [r for r in all_results if r['freeze_mode'] == 'frozen']
# unfrozen_results = [r for r in all_results if r['freeze_mode'] == 'unfrozen']

# def extract_metrics(results, ks):
#     accs, f1s = [], []
#     for k in ks:
#         match = next((r for r in results if r['k_value'] == k), None)
#         if match:
#             accs.append(match['test_accuracy'])
#             f1s.append(match['test_f1'])
#         else:
#             accs.append(None)
#             f1s.append(None)
#     return accs, f1s

# # Plot Accuracy
# frozen_ks = sorted(set(r['k_value'] for r in frozen_results))
# frozen_accs, frozen_f1s = extract_metrics(frozen_results, frozen_ks)
# unfrozen_ks = sorted(set(r['k_value'] for r in unfrozen_results))
# unfrozen_accs, unfrozen_f1s = extract_metrics(unfrozen_results, unfrozen_ks)

# axes[0].plot(frozen_ks, frozen_accs, 'o-', linewidth=2, markersize=8, label='Frozen Backbone')
# axes[0].plot(unfrozen_ks, unfrozen_accs, 's-', linewidth=2, markersize=8, label='Unfrozen Backbone')
# axes[0].axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='F1 Target (ref)')
# axes[0].set_xlabel('K (Samples per Class)', fontsize=12)
# axes[0].set_ylabel('Test Accuracy', fontsize=12)
# axes[0].set_title('Accuracy vs K Value', fontsize=13, fontweight='bold')
# axes[0].legend(fontsize=10)
# axes[0].grid(alpha=0.3)
# axes[0].set_xticks(K_VALUES)

# # Plot F1-Score with target line
# axes[1].plot(frozen_ks, frozen_f1s, 'o-', linewidth=2, markersize=8, label='Frozen Backbone')
# axes[1].plot(unfrozen_ks, unfrozen_f1s, 's-', linewidth=2, markersize=8, label='Unfrozen Backbone')
# axes[1].axhline(y=TARGET_F1_MIN, color='green', linestyle='--', linewidth=1.5, 
#                 label=f'Target F1 ≥ {TARGET_F1_MIN}')
# axes[1].set_xlabel('K (Samples per Class)', fontsize=12)
# axes[1].set_ylabel('F1-Score (Weighted)', fontsize=12)
# axes[1].set_title('F1-Score vs K Value', fontsize=13, fontweight='bold')
# axes[1].legend(fontsize=10)
# axes[1].grid(alpha=0.3)
# axes[1].set_xticks(K_VALUES)

# plt.tight_layout()
# plt.savefig(output_base / "comparison_plots.png", dpi=150, bbox_inches='tight')
# plt.close()

# # Save detailed per-class report for best run
# if best_test_preds is not None:
#     report = classification_report(best_test_labels, best_test_preds, 
#                                  target_names=[f"Class_{i}" for i in range(num_classes)],
#                                  output_dict=True, zero_division=0)
#     with open(output_base / "classification_report.json", 'w') as f:
#         json.dump(report, f, indent=2)

# print(f"\n✓ Results saved to {output_base}/")
# print(f"  - comparison_results.json")
# print(f"  - comparison_plots.png")
# print(f"  - classification_report.json (per-class metrics)")
# print(f"\n🎯 Remember: F1 ≥ {TARGET_F1_MIN} is your minimum target, not a stopping condition.")
# *************************************************************************************************