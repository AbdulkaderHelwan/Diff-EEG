#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import torch.optim as optim
from tqdm import tqdm
import json
from pathlib import Path

# ------------------------------------------------------------------
# 1. ENHANCED DIFFUSION SCHEDULER
# ------------------------------------------------------------------
class ImprovedDiffusionScheduler:
    def __init__(self, timesteps=1000, beta_schedule='linear', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def sample_random_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,), device=self.device).long()
    
    def add_noise(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

# ------------------------------------------------------------------
# 2. SINUSOIDAL POSITION EMBEDDINGS
# ------------------------------------------------------------------
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
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

# ------------------------------------------------------------------
# 3. ADVANCED RESIDUAL BLOCK WITH MULTI-SCALE ATTENTION
# ------------------------------------------------------------------
class AdvancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, use_attention=True, attention_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # Time embedding with better scaling
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
            nn.Dropout(dropout)
        )
        
        # Multi-scale normalization
        self.norm1 = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Advanced attention mechanism
        if use_attention:
            self.attn_norm = nn.InstanceNorm1d(out_channels, affine=True)
            self.attn_q = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_k = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_v = nn.Conv1d(out_channels, out_channels, 1)
            self.attn_proj = nn.Conv1d(out_channels, out_channels, 1)
            # Multi-head attention inspired approach
            self.attention_heads = attention_heads
            self.head_dim = out_channels // attention_heads
            assert self.head_dim * attention_heads == out_channels, "out_channels must be divisible by attention_heads"
        
    def forward(self, x, time_emb):
        # First normalization and conv
        B, C, L = x.shape
        h = self.norm1(x)
        
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time embedding conditioning
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.reshape(time_emb.shape[0], -1, 1)
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        # Second normalization and conv
        h = self.norm2(h)
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Self-attention block (multi-scale)
        if self.use_attention:
            attn_input = self.attn_norm(h)
            
            # Compute query, key, value
            q = self.attn_q(attn_input).view(B, self.attention_heads, self.head_dim, L)
            k = self.attn_k(attn_input).view(B, self.attention_heads, self.head_dim, L)
            v = self.attn_v(attn_input).view(B, self.attention_heads, self.head_dim, L)
            
            # Scaled dot-product attention
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1) + 1e-6  # Add epsilon for numerical stability
            
            attn_output = torch.einsum('bhqk,bhvd->bhqd', attn_weights, v)
            attn_output = attn_output.view(B, -1, L)  # Merge heads
            
            # Project back to original dimensions
            h = h + self.attn_proj(attn_output)
        
        # Residual connection
        residual = self.residual_conv(x)
        return h + residual

# ------------------------------------------------------------------
# 4. DEEP ENHANCED EEG DIFFUSION MODEL
# ------------------------------------------------------------------
class DeepEnhancedEEGDiffusionModel(nn.Module):
    def __init__(self, in_channels=22, model_channels=32, channel_multipliers=[1, 2, 4, 8],
                 num_res_blocks=2, time_emb_dim=512, dropout=0.1, attention_heads=8):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.attention_heads = attention_heads
        self.num_levels = len(channel_multipliers)
        
        # Enhanced time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Initial convolution with better initialization
        self.init_conv = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.init_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # ===== ENCODER =====
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        
        # Build encoder with increased depth
        for i, mult in enumerate(channel_multipliers):
            out_channels = model_channels * mult
            
            # Residual blocks at this resolution
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                # Place attention in later levels and at the end
                use_attention = (i >= len(channel_multipliers)//2) and (j == num_res_blocks - 1)
                block = AdvancedResidualBlock(
                    current_channels, out_channels, time_emb_dim, dropout, 
                    use_attention=use_attention, attention_heads=attention_heads
                )
                level_blocks.append(block)
                current_channels = out_channels
            
            self.down_blocks.append(level_blocks)
            
            # Downsampling layer (except at last level)
            if i != len(channel_multipliers) - 1:
                downsample = nn.Conv1d(current_channels, current_channels, 3, stride=2, padding=1)
                nn.init.kaiming_normal_(downsample.weight, mode='fan_out', nonlinearity='relu')
                self.down_blocks.append(nn.ModuleList([downsample]))
        
        # ===== BOTTLENECK =====
        # Deeper bottleneck with multiple layers
        self.bottleneck_blocks = nn.ModuleList([
            AdvancedResidualBlock(current_channels, current_channels, time_emb_dim, dropout, use_attention=True, attention_heads=attention_heads),
            AdvancedResidualBlock(current_channels, current_channels, time_emb_dim, dropout, use_attention=True, attention_heads=attention_heads),
            AdvancedResidualBlock(current_channels, current_channels, time_emb_dim, dropout, use_attention=True, attention_heads=attention_heads)
        ])
        
        # ===== DECODER =====
        self.up_blocks = nn.ModuleList()
        current_channels = model_channels * channel_multipliers[-1]  # Start with bottleneck channels
        
        # Build decoder with more sophisticated structure
        for i in range(len(channel_multipliers)):
            level_blocks = nn.ModuleList()
            mult = channel_multipliers[-(i+1)]
            out_channels = model_channels * mult
            
            # Upsampling layer (except at first level)
            if i > 0:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(current_channels, out_channels, 3, padding=1)
                )
                nn.init.kaiming_normal_(upsample[1].weight, mode='fan_out', nonlinearity='relu')
                level_blocks.append(upsample)
                current_channels = out_channels
            
            # Residual blocks at this resolution
            for j in range(num_res_blocks + 1):
                # All blocks take current_channels as input
                block_in_channels = current_channels
                
                # Place attention in earlier decoder levels
                use_attention = (i < len(channel_multipliers)//2) and (j == 0)
                block = AdvancedResidualBlock(
                    block_in_channels, out_channels, time_emb_dim, dropout,
                    use_attention=use_attention, attention_heads=attention_heads
                )
                level_blocks.append(block)
                current_channels = out_channels
            
            self.up_blocks.append(level_blocks)
        
        # Final layers with better initialization
        self.final_norm = nn.InstanceNorm1d(current_channels, affine=True)
        self.channel_fix_conv = nn.Conv1d(current_channels, model_channels, 1)  # Project to fixed channels
        self.final_conv = nn.Conv1d(model_channels, in_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        
        print(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        h = self.init_conv(x)
        
        # ===== ENCODER =====
        skips = []
        
        # Process encoder blocks
        for module_list in self.down_blocks:
            if len(module_list) == 1 and isinstance(module_list[0], nn.Conv1d):
                # This is a downsampling layer
                h = module_list[0](h)
            else:
                # Process residual blocks in this level
                for block in module_list:
                    if isinstance(block, AdvancedResidualBlock):
                        h = block(h, t_emb)
                        # Store skip connection from the last residual block of each level
                        if block == module_list[-1]:
                            skips.append(h)
                    else:
                        # This is a downsampling layer
                        h = block(h)
        
        # ===== BOTTLENECK =====
        for block in self.bottleneck_blocks:
            h = block(h, t_emb)
        
        # ===== DECODER =====
        skip_idx = len(skips) - 1
        
        # Process decoder blocks
        for module_list in self.up_blocks:
            skip_applied = False
            for block in module_list:
                if isinstance(block, nn.Sequential) and hasattr(block[0], 'scale_factor'):
                    # Upsampling layer
                    h = block(h)
                else:  # Residual block
                    if not skip_applied and skip_idx >= 0:
                        skip_tensor = skips[skip_idx]
                        skip_idx -= 1
                        if skip_tensor.shape[-1] != h.shape[-1]:
                            skip_tensor = F.interpolate(skip_tensor, size=h.shape[-1], mode='linear', align_corners=False)
                        h = h + skip_tensor
                        skip_applied = True
                    h = block(h, t_emb)
        
        # ===== FINAL LAYERS =====
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.channel_fix_conv(h)
        h = F.silu(h)
        return self.final_conv(h)

# ------------------------------------------------------------------
# 5. DATA LOADING
# ------------------------------------------------------------------
class BatchEEGDataset(Dataset):
    def __init__(
        self,
        batch_dir,
        seq_length=1280,
        normalize=True,
        max_batches=None,
        cache_in_memory=True,
        batch_files=None,
        normalization_stats=None,
        save_normalization=True,
    ):
        self.batch_dir = Path(batch_dir)
        self.seq_length = seq_length
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory
        self.save_normalization = save_normalization

        if batch_files is not None:
            self.batch_files = [Path(p) for p in batch_files]
        else:
            # Match both non_seizure_batch_*.npy and seizure_batch_*.npy,
            # but exclude *_labels.npy sidecar files.
            self.batch_files = sorted(
                f for f in self.batch_dir.glob("*.npy")
                if not f.name.endswith("_labels.npy")
            )
        if max_batches:
            self.batch_files = self.batch_files[:max_batches]
        
        if not self.batch_files:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        print(f"Found {len(self.batch_files)} batch files")
        
        self.sample_indices = []
        self.batch_data = []
        
        for batch_idx, batch_file in enumerate(self.batch_files):
            if self.cache_in_memory:
                data = np.load(batch_file, allow_pickle=True)
                self.batch_data.append(data)
                batch_size = len(data)
            else:
                self.batch_data.append(batch_file)
                with open(batch_file, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    else:
                        shape, _, _ = np.lib.format.read_array_header_2_0(f)
                    batch_size = shape[0]
            
            for sample_idx in range(batch_size):
                self.sample_indices.append((batch_idx, sample_idx))
        
        print(f"Total samples: {len(self.sample_indices)}")
        
        if self.normalize:
            if normalization_stats is None:
                self.compute_normalization_stats()
                if self.save_normalization:
                    # Save normalization in a fixed directory outside batch_dir
                    norm_dir = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/normalization/")
                    norm_dir.mkdir(exist_ok=True)
                    np.save(norm_dir / "mean.npy", self.mean)
                    np.save(norm_dir / "std.npy", self.std)
                    print(f"Saved normalization stats to {norm_dir}")
            else:
                provided_mean, provided_std = normalization_stats
                self.mean = np.array(provided_mean)
                self.std = np.array(provided_std)

            self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32)
            self.std_tensor = torch.tensor(self.std, dtype=torch.float32)
        
    def compute_normalization_stats(self, num_samples=None, chunk_size=1000):
        # Use all samples for normalization if num_samples is None
        total_samples = len(self.sample_indices)
        if num_samples is None or num_samples > total_samples:
            num_samples = total_samples
        sample_indices = np.arange(total_samples) if num_samples == total_samples else np.random.choice(total_samples, num_samples, replace=False)

        # Chunked computation
        n_channels = None
        n_seq = None
        sum_ = None
        sum_sq = None
        count = 0
        for i in tqdm(range(0, len(sample_indices), chunk_size), desc="Computing stats"):
            chunk_idxs = sample_indices[i:i+chunk_size]
            chunk_samples = []
            for idx in chunk_idxs:
                batch_idx, sample_idx = self.sample_indices[idx]
                sample = self.batch_data[batch_idx][sample_idx] if self.cache_in_memory else np.load(self.batch_data[batch_idx], allow_pickle=True)[sample_idx]
                chunk_samples.append(sample)
            chunk_samples = np.stack(chunk_samples)
            if n_channels is None:
                n_channels = chunk_samples.shape[1]
                n_seq = chunk_samples.shape[2]
                sum_ = np.zeros((n_channels, 1), dtype=np.float64)
                sum_sq = np.zeros((n_channels, 1), dtype=np.float64)
            # Compute sum and sum of squares for this chunk
            sum_ += np.mean(chunk_samples, axis=(0, 2))[:, np.newaxis] * len(chunk_samples)
            sum_sq += np.mean(chunk_samples ** 2, axis=(0, 2))[:, np.newaxis] * len(chunk_samples)
            count += len(chunk_samples)
        self.mean = sum_ / count
        self.std = np.sqrt((sum_sq / count) - (self.mean ** 2)) + 1e-6
        std_cap = np.percentile(self.std.flatten(), 95)
        self.std = np.clip(self.std, 0.1, std_cap)
        print(f"Mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}], Std range: [{self.std.min():.3f}, {self.std.max():.3f}]")
        
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        batch_idx, sample_idx = self.sample_indices[idx]
        sample = self.batch_data[batch_idx][sample_idx] if self.cache_in_memory else np.load(self.batch_data[batch_idx], allow_pickle=True)[sample_idx]
        
        if sample.shape[1] != self.seq_length:
            if sample.shape[1] > self.seq_length:
                start = np.random.randint(0, sample.shape[1] - self.seq_length)
                sample = sample[:, start:start + self.seq_length]
            else:
                pad_width = self.seq_length - sample.shape[1]
                sample = np.pad(sample, ((0, 0), (0, pad_width)), mode='edge')
        
        sample = torch.tensor(sample, dtype=torch.float32)
        if self.normalize:
            sample = (sample - self.mean_tensor) / self.std_tensor
        
        return {"eeg": sample}

    def get_normalization_stats(self):
        if not self.normalize:
            return None, None
        return self.mean.copy(), self.std.copy()

# ------------------------------------------------------------------
# 6. ENHANCED TRAINER WITH BETTER CONFIGURATION
# ------------------------------------------------------------------
class EnhancedDiffusionTrainer:
    def __init__(
        self,
        model,
        diffusion_scheduler,
        train_loader,
        val_loader,
        device,
        lr=5e-5,
        use_mixed_precision=True,
        recon_weight=0.1,
    ):
        
        self.model = model.to(device)
        self.diffusion = diffusion_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.recon_weight = recon_weight
        
        # Much more conservative optimizer settings
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6, betas=(0.9, 0.999))
        # Better learning rate scheduling
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=1e-7)
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
        
        self.step = 0
        self.best_val_loss = float('inf')
        self.output_dir = Path("/home/abdulh/scratch/training_diffusion2")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_loss(self, x, t, noise, pred_noise):
        # Clip predictions to prevent extreme values
        pred_noise = torch.clamp(pred_noise, -10, 10)
        
        noise_loss = F.mse_loss(pred_noise, noise)
        
        alpha = self.diffusion.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sigma = self.diffusion.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_recon = (x - sigma * pred_noise) / alpha
        x_recon = torch.clamp(x_recon, -3, 3)
        target_recon = (x - sigma * noise) / alpha
        recon_loss = F.mse_loss(x_recon, target_recon)
        
        total_loss = noise_loss + self.recon_weight * recon_loss
        return total_loss, {'noise_loss': noise_loss.item(), 'recon_loss': recon_loss.item()}
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = total_noise_loss = total_recon_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            x = batch["eeg"].to(self.device)
            batch_size = x.shape[0]
            t = self.diffusion.sample_random_timesteps(batch_size)
            noise = torch.randn_like(x)
            noisy_x, true_noise = self.diffusion.add_noise(x, t, noise)
            
            if self.use_mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_noise = self.model(noisy_x, t)
                    loss, loss_components = self.compute_loss(noisy_x, t, true_noise, pred_noise)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Very conservative gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_noise = self.model(noisy_x, t)
                loss, loss_components = self.compute_loss(noisy_x, t, true_noise, pred_noise)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            total_loss += loss.item()
            total_noise_loss += loss_components['noise_loss']
            total_recon_loss += loss_components['recon_loss']
            num_batches += 1
            self.step += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})
        
        return {'loss': total_loss / num_batches, 'noise_loss': total_noise_loss / num_batches, 'recon_loss': total_recon_loss / num_batches}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            x = batch["eeg"].to(self.device)
            batch_size = x.shape[0]
            
            # Test on more timesteps for better validation
            test_timesteps = [100, 250, 500, 750, 999]
            for t_val in test_timesteps:
                t = torch.full((batch_size,), t_val, device=self.device).long()
                noise = torch.randn_like(x)
                noisy_x, true_noise = self.diffusion.add_noise(x, t, noise)
                pred_noise = self.model(noisy_x, t)
                total_loss += F.mse_loss(pred_noise, true_noise).item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(best=True)
        
        return {'loss': avg_loss}
    
    def save_checkpoint(self, epoch=None, best=False):
        checkpoint = {
            'epoch': epoch, 'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        filename = self.output_dir / ("best_EEGDIFF2.pth" if best else f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")

# ------------------------------------------------------------------
# 7. MAIN TRAINING
# ------------------------------------------------------------------
def train_enhanced_diffusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BATCH_DATA_DIR = "/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/non_seizure_batches"
    
    all_batch_files = sorted(Path(BATCH_DATA_DIR).glob("non_seizure_batch_*.npy"))
    if not all_batch_files:
        print(f"No batch files found in {BATCH_DATA_DIR}")
        return
    
    split_idx = int(0.9 * len(all_batch_files))
    train_batches = all_batch_files[:split_idx]
    val_batches = all_batch_files[split_idx:]
    
    print(f"Training batches: {len(train_batches)}, Validation batches: {len(val_batches)}")
    
    # Load normalization stats from file
    norm_dir = Path("/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/non_seizure_batches/normalization")
    train_mean = np.load(norm_dir / "mean.npy")
    train_std = np.load(norm_dir / "std.npy")
    train_ds = BatchEEGDataset(
        BATCH_DATA_DIR,
        seq_length=1280,
        normalize=True,
        cache_in_memory=False,
        batch_files=train_batches,
        normalization_stats=(train_mean, train_std),
        save_normalization=False,
    )
    val_ds = BatchEEGDataset(
        BATCH_DATA_DIR,
        seq_length=1280,
        normalize=True,
        cache_in_memory=False,
        batch_files=val_batches,
        normalization_stats=(train_mean, train_std),
        save_normalization=False,
    )
    
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    # Increase batch size due to your computational resources
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    diffusion_scheduler = ImprovedDiffusionScheduler(timesteps=1000, beta_schedule='cosine', device=device)
    
    # Deep model configuration
    model = DeepEnhancedEEGDiffusionModel(
        in_channels=22,
        model_channels=32,
        channel_multipliers=[1, 2, 4, 8],
        num_res_blocks=2,
        time_emb_dim=512,
        dropout=0.1,
        attention_heads=8,
    )
    
    # Use lower learning rate for stability
    trainer = EnhancedDiffusionTrainer(
        model,
        diffusion_scheduler,
        train_loader,
        val_loader,
        device,
        lr=5e-5,
        use_mixed_precision=True,
        recon_weight=0.05,
    )
    
    print("\nStarting training...\n" + "=" * 60)
    
    history = {'train_loss': [], 'train_noise_loss': [], 'train_recon_loss': [], 'val_loss': [], 'learning_rate': []}
    
    # Train longer with better stopping criteria
    for epoch in range(1000):  # Much longer training
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate() if epoch % 2 == 0 else {'loss': history['val_loss'][-1] if history['val_loss'] else 0}
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_noise_loss'].append(train_metrics['noise_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch:04d}: Train={train_metrics['loss']:.6f}, Val={val_metrics['loss']:.6f}, Best Val={trainer.best_val_loss:.6f}")
        
        if epoch % 20 == 0:
            trainer.save_checkpoint(epoch=epoch, best=False)
        
        try:
            with open(trainer.output_dir / "training_history.json", 'w') as f:
                json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
        except:
            pass
        
        if epoch % 5 == 0:
            plot_training_progress(history, trainer.output_dir)
    
    print("\n" + "=" * 60 + "\nTraining completed!")
    return trainer, history

def plot_training_progress(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(len(history['train_loss']))
    
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', alpha=0.7)
    axes[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_noise_loss'], label='Noise', alpha=0.7)
    axes[0, 1].plot(epochs, history['train_recon_loss'], label='Recon', alpha=0.7)
    axes[0, 1].set(xlabel='Epoch', ylabel='Loss', title='Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['val_loss'], color='orange')
    axes[1, 0].set(xlabel='Epoch', ylabel='Loss', title='Validation Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['learning_rate'], color='purple')
    axes[1, 1].set(xlabel='Epoch', ylabel='LR', title='Learning Rate', yscale='log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_model_with_dummy_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model on {device}...\n")
    
    diffusion = ImprovedDiffusionScheduler(timesteps=1000, device=device)
    model = DeepEnhancedEEGDiffusionModel(
        in_channels=22, 
        model_channels=32, 
        channel_multipliers=[1, 2, 4, 8],
        num_res_blocks=2, 
        time_emb_dim=512, 
        dropout=0.1,
        attention_heads=8
    ).to(device)
    
    batch_size, seq_length = 4, 1280  # Reduced for testing
    dummy_input = torch.randn(batch_size, 22, seq_length, device=device)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input, dummy_t)
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful!")
        
        noisy_x, noise = diffusion.add_noise(dummy_input, dummy_t)
        pred_noise = model(noisy_x, dummy_t)
        loss = F.mse_loss(pred_noise, noise)
        print(f"Loss: {loss.item():.4f}")
        print("✓ Training step successful!")
        print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if test_model_with_dummy_data():
        print("\n" + "="*60)
        print("Model test passed! Starting training...")
        print("="*60 + "\n")
        
        try:
            trainer, history = train_enhanced_diffusion()
            print("\nTraining completed successfully!")
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nModel test failed! Fix the architecture issues first.")
