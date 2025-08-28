#!/usr/bin/env python3
"""
Comprehensive Model Visualization Script for SeqSetVAE
Includes: 
- Latent space visualization
- Reconstruction quality analysis
- Posterior collapse detection and visualization
- Real-time monitoring capabilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Try to import UMAP; fall back later if unavailable
try:
    import umap
except Exception:
    umap = None
import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path
from tqdm import tqdm

from model import SeqSetVAE, SeqSetVAEPretrain, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
import config

# Set matplotlib Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    """Configuration class for visualization"""
    def __init__(self):
        self.input_dim = config.input_dim
        self.reduced_dim = config.reduced_dim
        self.latent_dim = config.latent_dim
        self.levels = config.levels
        self.heads = config.heads
        self.m = config.m
        self.beta = config.beta
        self.lr = config.lr
        self.num_classes = config.num_classes
        self.ff_dim = config.ff_dim
        self.transformer_heads = config.transformer_heads
        self.transformer_layers = config.transformer_layers
        self.w = config.w
        self.free_bits = config.free_bits
        self.warmup_beta = config.warmup_beta
        self.max_beta = config.max_beta
        self.beta_warmup_steps = config.beta_warmup_steps
        self.kl_annealing = config.kl_annealing


class RealTimeCollapseVisualizer:
    """Real-time posterior collapse detection visualization tool"""
    
    def __init__(self, log_dir: str, update_interval: int = 1000):
        self.log_dir = log_dir
        self.update_interval = update_interval  # Update interval (ms)
        
        # Data storage
        self.max_points = 1000  # Maximum number of data points to display
        self.data = {
            'steps': deque(maxlen=self.max_points),
            'kl_divergence': deque(maxlen=self.max_points),
            'variance': deque(maxlen=self.max_points),
            'active_ratio': deque(maxlen=self.max_points),
            'recon_loss': deque(maxlen=self.max_points),
            'warnings': [],
            'collapse_detected': False,
            'collapse_step': None
        }
        
        # Thresholds (will be read from log files)
        self.thresholds = {
            'kl_threshold': 0.01,
            'var_threshold': 0.1,
            'active_threshold': 0.1
        }
        
        # Set up figure
        self.setup_figure()
        
        # State tracking
        self.last_modified = 0
        self.log_file_path = None
        
    def setup_figure(self):
        """Set up matplotlib figure"""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('VAE Posterior Collapse Real-time Monitoring', fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # KL divergence plot
        self.ax_kl = self.fig.add_subplot(gs[0, 0])
        self.ax_kl.set_title('KL Divergence Monitoring')
        self.ax_kl.set_ylabel('KL Divergence')
        self.ax_kl.grid(True, alpha=0.3)
        
        # Variance plot
        self.ax_var = self.fig.add_subplot(gs[0, 1])
        self.ax_var.set_title('Latent Variable Variance Monitoring')
        self.ax_var.set_ylabel('Mean Variance')
        self.ax_var.grid(True, alpha=0.3)
        
        # Active units ratio plot
        self.ax_active = self.fig.add_subplot(gs[1, 0])
        self.ax_active.set_title('Active Units Ratio Monitoring')
        self.ax_active.set_ylabel('Active Units Ratio')
        self.ax_active.grid(True, alpha=0.3)
        
        # Reconstruction loss plot
        self.ax_recon = self.fig.add_subplot(gs[1, 1])
        self.ax_recon.set_title('Reconstruction Loss Monitoring')
        self.ax_recon.set_ylabel('Reconstruction Loss')
        self.ax_recon.grid(True, alpha=0.3)
        
        # Status panel
        self.ax_status = self.fig.add_subplot(gs[2, :])
        self.ax_status.set_title('Detection Status')
        self.ax_status.axis('off')
        
        # Initialize lines
        self.line_kl, = self.ax_kl.plot([], [], 'b-', linewidth=2, label='KL Divergence')
        self.line_var, = self.ax_var.plot([], [], 'orange', linewidth=2, label='Variance')
        self.line_active, = self.ax_active.plot([], [], 'green', linewidth=2, label='Active Ratio')
        self.line_recon, = self.ax_recon.plot([], [], 'purple', linewidth=2, label='Recon Loss')
        
        # Threshold lines (will be added after data loading)
        self.threshold_lines = {}
        
    def find_log_file(self):
        """Find the latest log file"""
        if not os.path.exists(self.log_dir):
            return None
            
        log_files = []
        for file in os.listdir(self.log_dir):
            if file.endswith('.json'):
                log_files.append(os.path.join(self.log_dir, file))
        
        if not log_files:
            return None
            
        # Return the most recently modified file
        return max(log_files, key=os.path.getmtime)
    
    def parse_log_file(self, log_file: str):
        """Parse log file and extract metrics"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract metrics
                    if 'step' in data and 'metrics' in data:
                        step = data['step']
                        metrics = data['metrics']
                        
                        # Add to data storage
                        self.data['steps'].append(step)
                        self.data['kl_divergence'].append(metrics.get('kl_divergence', 0))
                        self.data['variance'].append(metrics.get('variance', 0))
                        self.data['active_ratio'].append(metrics.get('active_ratio', 0))
                        self.data['recon_loss'].append(metrics.get('reconstruction_loss', 0))
                        
                        # Check for collapse indicators
                        self.check_collapse_indicators(step, metrics)
                        
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            print(f"Error parsing log file: {e}")
    
    def check_collapse_indicators(self, step: int, metrics: Dict):
        """Check for posterior collapse indicators"""
        kl_div = metrics.get('kl_divergence', 0)
        variance = metrics.get('variance', 0)
        active_ratio = metrics.get('active_ratio', 0)
        
        warnings = []
        
        if kl_div < self.thresholds['kl_threshold']:
            warnings.append(f"KL divergence too low: {kl_div:.6f}")
            
        if variance < self.thresholds['var_threshold']:
            warnings.append(f"Variance too low: {variance:.6f}")
            
        if active_ratio < self.thresholds['active_threshold']:
            warnings.append(f"Active units ratio too low: {active_ratio:.3f}")
        
        if warnings and not self.data['collapse_detected']:
            self.data['collapse_detected'] = True
            self.data['collapse_step'] = step
            self.data['warnings'].extend(warnings)
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        # Check for new log data
        log_file = self.find_log_file()
        if log_file and log_file != self.log_file_path:
            self.log_file_path = log_file
            self.parse_log_file(log_file)
        
        # Update plot data
        if self.data['steps']:
            steps = list(self.data['steps'])
            kl_data = list(self.data['kl_divergence'])
            var_data = list(self.data['variance'])
            active_data = list(self.data['active_ratio'])
            recon_data = list(self.data['recon_loss'])
            
            # Update lines
            self.line_kl.set_data(steps, kl_data)
            self.line_var.set_data(steps, var_data)
            self.line_active.set_data(steps, active_data)
            self.line_recon.set_data(steps, recon_data)
            
            # Update axis limits
            for ax, data in [(self.ax_kl, kl_data), (self.ax_var, var_data), 
                           (self.ax_active, active_data), (self.ax_recon, recon_data)]:
                if data:
                    ax.relim()
                    ax.autoscale_view()
            
            # Update status panel
            self.update_status_panel()
        
        return self.line_kl, self.line_var, self.line_active, self.line_recon
    
    def update_status_panel(self):
        """Update the status panel"""
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        # Status text
        status_text = "Status: "
        if self.data['collapse_detected']:
            status_text += "⚠️ POSTERIOR COLLAPSE DETECTED"
            status_text += f"\nCollapse detected at step: {self.data['collapse_step']}"
            status_text += f"\nWarnings: {', '.join(self.data['warnings'])}"
        else:
            status_text += "✅ Training normally"
        
        if self.data['steps']:
            status_text += f"\nCurrent step: {self.data['steps'][-1]}"
            status_text += f"\nKL divergence: {self.data['kl_divergence'][-1]:.6f}"
            status_text += f"\nVariance: {self.data['variance'][-1]:.6f}"
            status_text += f"\nActive ratio: {self.data['active_ratio'][-1]:.3f}"
        
        self.ax_status.text(0.05, 0.5, status_text, fontsize=12, 
                           verticalalignment='center', transform=self.ax_status.transAxes)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print(f"🔍 Starting real-time monitoring of: {self.log_dir}")
        ani = animation.FuncAnimation(self.fig, self.update_plot, 
                                    interval=self.update_interval, blit=True)
        plt.show()


def load_model(checkpoint_path, config, checkpoint_type: str = 'auto'):
    """Load model from checkpoint with auto checkpoint-type detection.
    If the checkpoint was saved from pretraining (keys start with 'set_encoder.'),
    we instantiate SeqSetVAEPretrain. Otherwise we use SeqSetVAE.
    """
    print(f"Loading model weights from {checkpoint_path}")

    # Load raw state dict (weights only)
    state_dict = load_checkpoint_weights(checkpoint_path, device='cpu')

    # Auto-detect checkpoint type if requested
    if checkpoint_type == 'auto':
        if any(k.startswith('set_encoder.') for k in state_dict.keys()):
            checkpoint_type = 'pretrain'
        else:
            checkpoint_type = 'finetune'
    print(f"Detected/selected checkpoint type: {checkpoint_type}")

    # Build the appropriate model
    if checkpoint_type == 'pretrain':
        model = SeqSetVAEPretrain(
            input_dim=config.input_dim,
            reduced_dim=config.reduced_dim,
            latent_dim=config.latent_dim,
            levels=config.levels,
            heads=config.heads,
            m=config.m,
            beta=config.beta,
            lr=config.lr,
            ff_dim=config.ff_dim,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
            warmup_beta=config.warmup_beta,
            max_beta=config.max_beta,
            beta_warmup_steps=config.beta_warmup_steps,
            free_bits=config.free_bits,
        )
    else:
        model = SeqSetVAE(
            input_dim=config.input_dim,
            reduced_dim=config.reduced_dim,
            latent_dim=config.latent_dim,
            levels=config.levels,
            heads=config.heads,
            m=config.m,
            beta=config.beta,
            lr=config.lr,
            num_classes=config.num_classes,
            ff_dim=config.ff_dim,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
            freeze_ratio=0.0,
            pretrained_ckpt=None,
            w=config.w,
            free_bits=config.free_bits,
            warmup_beta=config.warmup_beta,
            max_beta=config.max_beta,
            beta_warmup_steps=config.beta_warmup_steps,
            kl_annealing=config.kl_annealing,
        )

    # Load parameters (non-strict so extra classifier heads etc. won't block)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)} parameters not loaded")
        if len(missing_keys) <= 5:
            for key in missing_keys[:5]:
                print(f"   - {key}")
    else:
        print("✅ All model parameters loaded successfully")

    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)} extra parameters ignored")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys[:5]:
                print(f"   - {key}")

    return model


def collect_single_sample_representations(model, dataloader, sample_idx=0):
    """Collect latent representations for a single patient sample using current model API.
    Expects dataloader batches to be dicts with keys: 'var', 'val', 'minute', 'set_id', 'label' (optional), 'padding_mask' (optional).
    """
    print(f"Collecting representations for sample {sample_idx}")
    model.eval()
    device = next(model.parameters()).device

    # Helper to move a tensor dict to device
    def to_device(batch_dict):
        out = {}
        for k, v in batch_dict.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    # Locate the target sample (with batch_size=1, index==i)
    target_batch = None
    for i, batch_data in enumerate(dataloader):
        if i == sample_idx:
            target_batch = batch_data
            break

    if target_batch is None:
        raise ValueError(f"Sample {sample_idx} not found in dataloader")

    batch_on_device = to_device(target_batch)

    with torch.no_grad():
        # Forward pass (SeqSetVAE.forward accepts a dict batch)
        _ = model(batch_on_device)
        # Retrieve latent variables collected during forward
        z_source = None
        if hasattr(model, '_last_z_list') and model._last_z_list:
            z_source = model._last_z_list
        elif hasattr(model, 'setvae') and hasattr(model.setvae, '_last_z_list'):
            z_source = model.setvae._last_z_list
        if not z_source:
            raise RuntimeError("Latent variables not found on model after forward pass")

        # z_source is a list of tuples: (z_sample, mu, logvar) per level
        z_samples = [z_sample.detach().cpu().numpy() for (z_sample, mu, logvar) in z_source]
        mus = [mu.detach().cpu().numpy() for (z_sample, mu, logvar) in z_source]
        logvars = [logvar.detach().cpu().numpy() for (z_sample, mu, logvar) in z_source]

    results = {
        'z_samples': z_samples,
        'mus': mus,
        'logvars': logvars,
    }
    # Try to fetch reconstructions and targets from model for later visualization
    try:
        if hasattr(model, '_last_recon_cat') and model._last_recon_cat is not None:
            results['recon_events'] = model._last_recon_cat.detach().cpu().numpy()
        if hasattr(model, '_last_target_cat') and model._last_target_cat is not None:
            results['orig_events'] = model._last_target_cat.detach().cpu().numpy()
    except Exception:
        pass
    return results


def analyze_single_sample_posterior_collapse(mus, logvars, threshold=0.01):
    """Analyze posterior collapse indicators using mu/logvar from latent levels."""
    print("Analyzing posterior collapse indicators...")
    collapse_analysis = {}

    for level, (mu, logvar) in enumerate(zip(mus, logvars)):
        # Shapes are [B, 1, D]; use the first (and only) element
        mu_ = mu[0]
        logvar_ = logvar[0]
        # KL divergence to N(0, I) per dimension, averaged
        kl_div = 0.5 * np.sum(mu_**2 + np.exp(logvar_) - logvar_ - 1, axis=-1)
        mean_kl = float(np.mean(kl_div))

        variance = np.exp(logvar_)
        mean_variance = float(np.mean(variance))
        active_units = np.sum(variance > threshold, axis=-1)
        active_ratio = float(np.mean(active_units) / variance.shape[-1])

        collapse_analysis[f'level_{level}'] = {
            'kl_divergence': mean_kl,
            'variance': mean_variance,
            'active_ratio': active_ratio,
            'collapse_risk': 'high' if mean_kl < threshold or active_ratio < 0.1 else 'low'
        }

    return collapse_analysis


def _plot_input_vs_recon_umap(orig_events: np.ndarray, recon_events: np.ndarray, save_dir: str, sample_idx: int):
    """Project original vs reconstructed events to 2D and overlay scatter.
    Uses UMAP if available; otherwise falls back to PCA.
    """
    if orig_events is None or recon_events is None:
        print("⚠️  Skipping UMAP overlay: missing original or reconstructed events")
        return
    # Squeeze batch dim if present
    if orig_events.ndim == 3:
        orig_events = orig_events[0]
    if recon_events.ndim == 3:
        recon_events = recon_events[0]
    # Align lengths (in case of slight mismatch)
    n = min(len(orig_events), len(recon_events))
    orig = orig_events[:n]
    recon = recon_events[:n]
    # Optionally subsample to keep visualization responsive
    max_points = 10000
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        orig = orig[idx]
        recon = recon[idx]
        n = max_points
    # Labels for plotting
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)], axis=0)
    # Standardize using ORIGINAL data only; then apply the same transform to recon
    try:
        scaler = StandardScaler(with_mean=True, with_std=True).fit(orig)
        orig_scaled = scaler.transform(orig)
        recon_scaled = scaler.transform(recon)
    except Exception:
        orig_scaled = orig
        recon_scaled = recon
    # Fit reducer on ORIGINAL only, then transform RECON into the same space
    try:
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            orig_emb = reducer.fit_transform(orig_scaled)
            recon_emb = reducer.transform(recon_scaled)
        else:
            pca = PCA(n_components=2)
            orig_emb = pca.fit_transform(orig_scaled)
            recon_emb = pca.transform(recon_scaled)
    except Exception:
        pca = PCA(n_components=2)
        orig_emb = pca.fit_transform(orig_scaled)
        recon_emb = pca.transform(recon_scaled)
    emb = np.vstack([orig_emb, recon_emb])
    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=8, alpha=0.7, c='tab:blue', label='Original')
    ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=8, alpha=0.7, c='tab:orange', label='Reconstruction')
    # Report quantitative alignment in title
    try:
        # compute mean nearest-neighbor cosine similarity and l2 distance in original feature space
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(orig_scaled)
        dists, idxs = nn.kneighbors(recon_scaled)
        chamfer_like = float(np.mean(dists))
        title_extra = f"  |  NN-dist(mean)={chamfer_like:.3f}"
    except Exception:
        title_extra = ""
    ax.set_title('Events: Original vs Reconstruction (UMAP/PCA)'+title_extra)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f'events_umap_overlay_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ UMAP overlay saved to: {save_path}")
    plt.close(fig)


def plot_single_sample_visualizations(results, save_dir, sample_idx=0):
    """Create visualizations for a single sample using latent mus/logvars only."""
    print(f"Creating visualizations for sample {sample_idx}")
    os.makedirs(save_dir, exist_ok=True)

    mus = results['mus']
    logvars = results['logvars']
    collapse_analysis = analyze_single_sample_posterior_collapse(mus, logvars)

    # Optional: UMAP overlay of original vs reconstructed events
    _plot_input_vs_recon_umap(
        results.get('orig_events', None),
        results.get('recon_events', None),
        save_dir,
        sample_idx
    )

    # Figure layout (3 rows x 2 cols)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Latent space visualization (t-SNE on mu)
    ax1 = fig.add_subplot(gs[0, 0])
    for level, mu in enumerate(mus):
        mu_ = mu[0]
        if mu_.shape[-1] > 2:
            n_samples = mu_.shape[0]
            if n_samples < 2:
                if mu_.shape[1] >= 2:
                    mu_2d = mu_[:, :2]
                else:
                    mu_2d = np.hstack([mu_, np.zeros((n_samples, 2 - mu_.shape[1]))])
            elif n_samples < 5:
                try:
                    mu_2d = PCA(n_components=2).fit_transform(mu_)
                except Exception:
                    if mu_.shape[1] >= 2:
                        mu_2d = mu_[:, :2]
                    else:
                        mu_2d = np.hstack([mu_, np.zeros((n_samples, 2 - mu_.shape[1]))])
            else:
                safe_perplexity = min(30.0, max(5.0, float(n_samples - 1) / 3.0))
                if safe_perplexity >= n_samples:
                    safe_perplexity = float(n_samples - 1)
                    if safe_perplexity < 2.0:
                        safe_perplexity = 2.0
                try:
                    tsne = TSNE(
                        n_components=2,
                        random_state=42,
                        perplexity=safe_perplexity,
                        init="random",
                        learning_rate="auto",
                    )
                    mu_2d = tsne.fit_transform(mu_)
                except Exception:
                    try:
                        mu_2d = PCA(n_components=2).fit_transform(mu_)
                    except Exception:
                        if mu_.shape[1] >= 2:
                            mu_2d = mu_[:, :2]
                        else:
                            mu_2d = np.hstack([mu_, np.zeros((n_samples, 2 - mu_.shape[1]))])
        else:
            mu_2d = mu_
        ax1.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.7, label=f'Level {level}')
    ax1.set_title('Latent Means (t-SNE)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Posterior collapse indicators
    ax2 = fig.add_subplot(gs[0, 1])
    levels = list(collapse_analysis.keys())
    kl_values = [collapse_analysis[level]['kl_divergence'] for level in levels]
    var_values = [collapse_analysis[level]['variance'] for level in levels]
    active_values = [collapse_analysis[level]['active_ratio'] for level in levels]
    x = np.arange(len(levels))
    width = 0.25
    ax2.bar(x - width, kl_values, width, label='KL Divergence', alpha=0.8)
    ax2.bar(x, var_values, width, label='Variance', alpha=0.8)
    ax2.bar(x + width, active_values, width, label='Active Ratio', alpha=0.8)
    ax2.set_title('Posterior Collapse Indicators')
    ax2.set_xlabel('Latent Level')
    ax2.set_ylabel('Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{i}' for i in range(len(levels))])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Latent variable distributions (mu)
    ax3 = fig.add_subplot(gs[1, :])
    for level, mu in enumerate(mus):
        ax3.hist(mu[0].flatten(), bins=50, alpha=0.6, label=f'Level {level}', density=True)
    ax3.set_title('Latent Mean Distributions')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Variance profile per level
    ax4 = fig.add_subplot(gs[2, 0])
    for level, logvar in enumerate(logvars):
        variance = np.exp(logvar[0])
        ax4.plot(variance.mean(axis=0), alpha=0.8, label=f'Level {level}')
    ax4.set_title('Latent Variance Profile (mean per dim)')
    ax4.set_xlabel('Latent Dimension')
    ax4.set_ylabel('Variance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Correlation matrix of concatenated mus
    ax5 = fig.add_subplot(gs[2, 1])
    if len(mus) > 0:
        # Estimate correlation via Monte Carlo samples from q(z|x) to avoid empty heatmap with batch_size=1
        num_mc_samples = 512
        samples_list = []
        for mu, logvar in zip(mus, logvars):
            mu_vec = np.asarray(mu[0]).squeeze(0)
            logvar_vec = np.asarray(logvar[0]).squeeze(0)
            std_vec = np.exp(0.5 * logvar_vec)
            std_vec = np.maximum(std_vec, 1e-6)
            samples = np.random.normal(loc=mu_vec, scale=std_vec, size=(num_mc_samples, mu_vec.shape[-1]))
            samples_list.append(samples)
        sample_matrix = np.concatenate(samples_list, axis=1)
        # compute correlation across dimensions
        corr_matrix = np.corrcoef(sample_matrix, rowvar=False)
        # sanitize possible NaNs/Infs
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax5.set_title('Latent Mean Correlations')
        plt.colorbar(im, ax=ax5)

    # Save
    save_path = os.path.join(save_dir, f'comprehensive_analysis_sample_{sample_idx}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive visualization saved to: {save_path}")
    plt.show()

    return collapse_analysis


def create_dashboard(log_dir: str, update_interval: int = 1000):
    """Create a real-time monitoring dashboard"""
    visualizer = RealTimeCollapseVisualizer(log_dir, update_interval)
    visualizer.start_monitoring()


def _collect_mu_var_with_labels(model, dataloader, max_samples=100):
    """Collect posterior mean and variance per patient with labels.
    Assumes dataloader yields batches where each batch corresponds to a single patient
    (batch_size=1). Uses model._last_z_list to read mu/logvar from the VAE for the
    first set of the patient, taking the last latent level.
    Returns a list of dicts with keys: 'mu' (np.ndarray[D]), 'var' (np.ndarray[D]), 'label' (int).
    """
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    results = []
    with torch.no_grad():
        for batch in dataloader:
            # Ensure batch tensors on device
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v

            # Forward pass; model accepts dict batch
            try:
                _ = model(batch_on_device)
            except Exception:
                # Some models may require tuple signature; try fallback paths
                if {'var', 'val'}.issubset(batch_on_device.keys()):
                    _ = model(batch_on_device['var'].to(device), batch_on_device['val'].to(device))
                else:
                    continue

            # Read mu/logvar list stored on model
            if not hasattr(model, '_last_z_list') or model._last_z_list is None:
                continue
            try:
                # Take last latent level from first set: (z_sample, mu, logvar)
                z_sample, mu, logvar = model._last_z_list[-1]
                # Shapes: [B, 1, D]; squeeze set dim
                mu = mu.squeeze(1)
                logvar = logvar.squeeze(1)
                # Expect B == 1
                mu_np = mu[0].detach().cpu().numpy()
                var_np = torch.exp(logvar[0]).detach().cpu().numpy()
            except Exception:
                continue

            label = int(batch_on_device.get('label', torch.tensor([0], device=device))[0].item())
            results.append({'mu': mu_np, 'var': var_np, 'label': label})
            if len(results) >= max_samples:
                break
    return results


def _plot_mu_var_per_sample_grid(samples, num_pos=4, num_neg=4, save_path=None):
    """Plot per-sample 2D scatter (x=mu_d, y=var_d) for a grid of pos/neg examples."""
    # Select examples
    pos_examples = [s for s in samples if s['label'] == 1][:num_pos]
    neg_examples = [s for s in samples if s['label'] == 0][:num_neg]
    total = len(pos_examples) + len(neg_examples)
    if total == 0:
        print("No samples to plot.")
        return

    rows = 2 if (pos_examples and neg_examples) else 1
    cols = max(len(pos_examples), len(neg_examples))
    cols = max(cols, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), squeeze=False)

    # Helper to plot a single panel
    def _plot_panel(ax, sample, title):
        mu = sample['mu']
        var = sample['var']
        ax.scatter(mu, var, s=12, alpha=0.8, c='tab:orange' if sample['label'] == 1 else 'tab:blue')
        ax.axvline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('mean (mu)')
        ax.set_ylabel('variance (exp(logvar))')
        ax.set_title(title)
        # Optional: show summary stats in corner
        try:
            corr = np.corrcoef(mu, var)[0, 1]
            ax.text(0.02, 0.98, f"r={corr:.2f}", transform=ax.transAxes, va='top', ha='left', fontsize=9)
        except Exception:
            pass

    # Plot pos row
    for j in range(cols):
        ax = axes[0, j]
        if j < len(pos_examples):
            _plot_panel(ax, pos_examples[j], f"pos sample #{j}")
        else:
            ax.axis('off')

    # Plot neg row if present
    if rows == 2:
        for j in range(cols):
            ax = axes[1, j]
            if j < len(neg_examples):
                _plot_panel(ax, neg_examples[j], f"neg sample #{j}")
            else:
                ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved per-sample mean-var grid to: {save_path}")
    plt.show()


def _plot_mu_var_aggregate(samples, save_path=None):
    """Aggregate all dimensions from all samples and plot mean-vs-variance distribution by label."""
    if not samples:
        print("No samples to aggregate.")
        return
    mu_pos = []
    var_pos = []
    mu_neg = []
    var_neg = []
    for s in samples:
        if s['label'] == 1:
            mu_pos.append(s['mu'])
            var_pos.append(s['var'])
        else:
            mu_neg.append(s['mu'])
            var_neg.append(s['var'])
    if mu_pos:
        mu_pos = np.concatenate(mu_pos)
        var_pos = np.concatenate(var_pos)
    else:
        mu_pos = np.array([])
        var_pos = np.array([])
    if mu_neg:
        mu_neg = np.concatenate(mu_neg)
        var_neg = np.concatenate(var_neg)
    else:
        mu_neg = np.array([])
        var_neg = np.array([])

    fig, ax = plt.subplots(figsize=(6, 5))
    if mu_neg.size > 0:
        ax.scatter(mu_neg, var_neg, s=6, alpha=0.25, c='tab:blue', label='neg dims')
    if mu_pos.size > 0:
        ax.scatter(mu_pos, var_pos, s=6, alpha=0.25, c='tab:orange', label='pos dims')
    ax.axvline(0.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('mean (mu)')
    ax.set_ylabel('variance (exp(logvar))')
    ax.set_title('Aggregate mean vs variance by label (dims pooled)')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved aggregate mean-var distribution to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Visualization')
    parser.add_argument('--mode', type=str, choices=['static', 'realtime', 'meanvar'], default='static',
                       help='Visualization mode: static analysis or real-time monitoring')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--checkpoint_type', type=str, choices=['auto', 'pretrain', 'finetune'], default='auto',
                       help='Type of checkpoint to load (auto-detect by default)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory (patient_ehr folder)')
    parser.add_argument('--params_map_path', type=str, required=True,
                       help='Path to stats.csv for value normalization')
    parser.add_argument('--label_path', type=str, required=True,
                       help='Path to oc.csv with labels')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to analyze (for static mode)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory for real-time monitoring')
    parser.add_argument('--update_interval', type=int, default=1000,
                       help='Update interval in milliseconds (for real-time mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'static':
        # Static analysis mode
        print("🔍 Running static analysis mode...")
        
        # Load model
        model_config = Config()
        model = load_model(args.checkpoint, model_config, checkpoint_type=args.checkpoint_type)
        
        # Load data
        data_module = SeqSetVAEDataModule(
            saved_dir=args.data_dir,
            params_map_path=args.params_map_path,
            label_path=args.label_path,
            batch_size=1
        )
        data_module.num_workers = 0
        data_module.pin_memory = False
        data_module.setup()
        
        # Collect representations
        results = collect_single_sample_representations(model, data_module.train_dataloader(), args.sample_idx)
        
        # Create visualizations
        collapse_analysis = plot_single_sample_visualizations(results, args.save_dir, args.sample_idx)
        
        # Print summary
        print("\n📊 Analysis Summary:")
        for level, analysis in collapse_analysis.items():
            print(f"  {level}:")
            print(f"    - KL Divergence: {analysis['kl_divergence']:.6f}")
            print(f"    - Variance: {analysis['variance']:.6f}")
            print(f"    - Active Ratio: {analysis['active_ratio']:.3f}")
            print(f"    - Collapse Risk: {analysis['collapse_risk']}")
        
    elif args.mode == 'realtime':
        # Real-time monitoring mode
        print("🔍 Running real-time monitoring mode...")
        create_dashboard(args.log_dir, args.update_interval)
    elif args.mode == 'meanvar':
        print("🔍 Running mean/variance scatter visualization...")
        # Load model
        model_config = Config()
        model = load_model(args.checkpoint, model_config, checkpoint_type=args.checkpoint_type)
        # Load data (batch_size=1 for per-patient extraction)
        data_module = SeqSetVAEDataModule(
            saved_dir=args.data_dir,
            params_map_path=args.params_map_path,
            label_path=args.label_path,
            batch_size=1
        )
        data_module.num_workers = 0
        data_module.pin_memory = False
        data_module.setup()
        dl = data_module.train_dataloader()
        # Collect mu/var
        samples = _collect_mu_var_with_labels(model, dl, max_samples=200)
        if not samples:
            print("⚠️ No mu/var samples collected. Check dataloader and checkpoint compatibility.")
            return
        # Plot per-sample grid
        os.makedirs(args.save_dir, exist_ok=True)
        grid_path = os.path.join(args.save_dir, 'meanvar_grid.png')
        _plot_mu_var_per_sample_grid(samples, num_pos=6, num_neg=6, save_path=grid_path)
        # Plot aggregated distribution
        agg_path = os.path.join(args.save_dir, 'meanvar_aggregate.png')
        _plot_mu_var_aggregate(samples, save_path=agg_path)


if __name__ == "__main__":
    main()
