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
# import umap  # optional
import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path
from tqdm import tqdm

from model import SeqSetVAE
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


def _prepare_state_dict_for_seqsetvae(state_dict):
    """Clean prefixes and remap keys if checkpoint comes from pretrain module.
    - Strip leading 'model.' / 'module.'
    - If keys start with 'set_encoder.', map to 'setvae.setvae.' for SeqSetVAE
    - Keep transformer/decoder/time-encoding related keys as-is
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned[new_k] = v

    keys = list(cleaned.keys())
    is_pretrain_ckpt = any(k.startswith("set_encoder.") for k in keys)
    if not is_pretrain_ckpt:
        return cleaned

    remapped = {}
    for k, v in cleaned.items():
        if k.startswith('set_encoder.'):
            remapped['setvae.setvae.' + k[len('set_encoder.'):]] = v
        elif (
            k.startswith('transformer.') or
            k.startswith('post_transformer_norm.') or
            k.startswith('decoder.') or
            k.startswith('rel_time_bucket_embed.') or
            k in ('alibi_slope', 'time_tau', 'time_bucket_edges')
        ):
            remapped[k] = v
        else:
            # Skip unrelated keys (e.g., metrics buffers from different modules)
            pass
    print("🔁 Detected pretrain checkpoint; remapped keys for SeqSetVAE.")
    return remapped


def load_model(checkpoint_path, config):
    """Load model from checkpoint - only loads weights since only weights are saved"""
    print(f"Loading model weights from {checkpoint_path}")
    
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
        pretrained_ckpt=None,  # Don't load pretrained here
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
    )
    
    # Import the utility function from model.py
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import load_checkpoint_weights
    
    # Load checkpoint weights using utility function
    state_dict = load_checkpoint_weights(checkpoint_path, device='cpu')
    
    # Clean/remap keys if needed (e.g., pretrain -> finetune)
    state_dict = _prepare_state_dict_for_seqsetvae(state_dict)
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)} parameters not loaded")
        if len(missing_keys) <= 5:  # Show first few missing keys
            for key in missing_keys[:5]:
                print(f"   - {key}")
    else:
        print("✅ All model parameters loaded successfully")
    
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)} extra parameters ignored")
        if len(unexpected_keys) <= 5:  # Show first few unexpected keys
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


def plot_single_sample_visualizations(results, save_dir, sample_idx=0):
    """Create visualizations for a single sample using latent mus/logvars only."""
    print(f"Creating visualizations for sample {sample_idx}")
    os.makedirs(save_dir, exist_ok=True)

    mus = results['mus']
    logvars = results['logvars']
    collapse_analysis = analyze_single_sample_posterior_collapse(mus, logvars)

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
        mu_combined = np.concatenate([m[0] for m in mus], axis=1)
        corr_matrix = np.corrcoef(mu_combined.T)
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
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


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Visualization')
    parser.add_argument('--mode', type=str, choices=['static', 'realtime'], default='static',
                       help='Visualization mode: static analysis or real-time monitoring')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
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
        model = load_model(args.checkpoint, model_config)
        
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


if __name__ == "__main__":
    main()