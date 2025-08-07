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
import umap
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
    """Collect latent representations for a single sample"""
    print(f"Collecting representations for sample {sample_idx}")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get the specific sample
    batch = None
    for i, batch_data in enumerate(dataloader):
        if i == sample_idx // dataloader.batch_size:
            batch = batch_data
            break
    
    if batch is None:
        raise ValueError(f"Sample {sample_idx} not found in dataloader")
    
    var, val, labels = batch
    var = var.to(device)
    val = val.to(device)
    
    # Get sample within batch
    sample_in_batch = sample_idx % dataloader.batch_size
    var_sample = var[sample_in_batch:sample_in_batch+1]
    val_sample = val[sample_in_batch:sample_in_batch+1]
    
    with torch.no_grad():
        # Forward pass
        recon, z_list, z_means, z_logvars, classification_output = model(var_sample, val_sample)
        
        # Extract latent representations
        latent_repr = {
            'z_means': [z_mean.cpu().numpy() for z_mean in z_means],
            'z_logvars': [z_logvar.cpu().numpy() for z_logvar in z_logvars],
            'z_list': [z.cpu().numpy() for z in z_list],
            'recon': recon.cpu().numpy(),
            'original': val_sample.cpu().numpy(),
            'classification': classification_output.cpu().numpy()
        }
    
    return latent_repr


def analyze_single_sample_posterior_collapse(z_means, z_logvars, threshold=0.01):
    """Analyze posterior collapse for a single sample"""
    print("Analyzing posterior collapse indicators...")
    
    collapse_analysis = {}
    
    for level, (z_mean, z_logvar) in enumerate(zip(z_means, z_logvars)):
        # Calculate KL divergence (simplified)
        kl_div = 0.5 * np.sum(z_mean**2 + np.exp(z_logvar) - z_logvar - 1, axis=-1)
        mean_kl = np.mean(kl_div)
        
        # Calculate variance
        variance = np.exp(z_logvar)
        mean_variance = np.mean(variance)
        
        # Calculate active units ratio
        active_units = np.sum(variance > threshold, axis=-1)
        active_ratio = np.mean(active_units) / variance.shape[-1]
        
        collapse_analysis[f'level_{level}'] = {
            'kl_divergence': mean_kl,
            'variance': mean_variance,
            'active_ratio': active_ratio,
            'collapse_risk': 'high' if mean_kl < threshold or active_ratio < 0.1 else 'low'
        }
    
    return collapse_analysis


def plot_single_sample_visualizations(results, save_dir, sample_idx=0):
    """Create comprehensive visualizations for a single sample"""
    print(f"Creating visualizations for sample {sample_idx}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    z_means = results['z_means']
    z_logvars = results['z_logvars']
    z_list = results['z_list']
    recon = results['recon']
    original = results['original']
    collapse_analysis = analyze_single_sample_posterior_collapse(z_means, z_logvars)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Latent space visualization (t-SNE)
    ax1 = fig.add_subplot(gs[0, 0])
    for level, z_mean in enumerate(z_means):
        if z_mean.shape[-1] > 2:
            # Use t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            z_2d = tsne.fit_transform(z_mean[0])
        else:
            z_2d = z_mean[0]
        
        ax1.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, label=f'Level {level}')
    
    ax1.set_title('Latent Space Visualization (t-SNE)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Posterior collapse analysis
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
    
    # 3. Reconstruction quality
    ax3 = fig.add_subplot(gs[0, 2])
    original_flat = original[0].flatten()
    recon_flat = recon[0].flatten()
    
    ax3.scatter(original_flat, recon_flat, alpha=0.6)
    ax3.plot([original_flat.min(), original_flat.max()], 
             [original_flat.min(), original_flat.max()], 'r--', alpha=0.8)
    ax3.set_title('Reconstruction Quality')
    ax3.set_xlabel('Original Values')
    ax3.set_ylabel('Reconstructed Values')
    ax3.grid(True, alpha=0.3)
    
    # 4. Latent variable distributions
    ax4 = fig.add_subplot(gs[1, :])
    for level, z_mean in enumerate(z_means):
        ax4.hist(z_mean[0].flatten(), bins=50, alpha=0.7, 
                label=f'Level {level}', density=True)
    
    ax4.set_title('Latent Variable Distributions')
    ax4.set_xlabel('Latent Variable Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Variance analysis
    ax5 = fig.add_subplot(gs[2, 0])
    for level, z_logvar in enumerate(z_logvars):
        variance = np.exp(z_logvar[0])
        ax5.plot(variance.mean(axis=0), alpha=0.7, label=f'Level {level}')
    
    ax5.set_title('Latent Variable Variance')
    ax5.set_xlabel('Latent Dimension')
    ax5.set_ylabel('Variance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Correlation matrix
    ax6 = fig.add_subplot(gs[2, 1])
    if len(z_means) > 0:
        z_combined = np.concatenate([z_mean[0] for z_mean in z_means], axis=1)
        corr_matrix = np.corrcoef(z_combined.T)
        im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax6.set_title('Latent Variable Correlations')
        plt.colorbar(im, ax=ax6)
    
    # 7. Training dynamics summary
    ax7 = fig.add_subplot(gs[2, 2])
    collapse_risks = [collapse_analysis[level]['collapse_risk'] for level in levels]
    risk_counts = {'high': collapse_risks.count('high'), 'low': collapse_risks.count('low')}
    
    colors = ['red' if risk == 'high' else 'green' for risk in collapse_risks]
    ax7.bar(range(len(levels)), [1]*len(levels), color=colors, alpha=0.7)
    ax7.set_title('Collapse Risk Assessment')
    ax7.set_xlabel('Latent Level')
    ax7.set_ylabel('Risk Level')
    ax7.set_xticks(range(len(levels)))
    ax7.set_xticklabels([f'L{i}' for i in range(len(levels))])
    ax7.set_ylim(0, 1.2)
    
    # Add text annotations
    for i, risk in enumerate(collapse_risks):
        ax7.text(i, 0.5, risk.upper(), ha='center', va='center', 
                fontweight='bold', color='white')
    
    # 8. Detailed metrics table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create detailed metrics table
    table_data = []
    for level in levels:
        analysis = collapse_analysis[level]
        table_data.append([
            level,
            f"{analysis['kl_divergence']:.6f}",
            f"{analysis['variance']:.6f}",
            f"{analysis['active_ratio']:.3f}",
            analysis['collapse_risk']
        ])
    
    table = ax8.table(cellText=table_data,
                     colLabels=['Level', 'KL Divergence', 'Variance', 'Active Ratio', 'Risk'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the risk column
    for i, row in enumerate(table_data):
        if row[4] == 'high':
            table[(i+1, 4)].set_facecolor('red')
        else:
            table[(i+1, 4)].set_facecolor('green')
    
    plt.suptitle(f'Comprehensive Model Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    # Save the comprehensive visualization
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
                       help='Path to data directory')
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
            data_dir=args.data_dir,
            batch_size=1,  # Use batch size 1 for single sample analysis
            num_workers=0
        )
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