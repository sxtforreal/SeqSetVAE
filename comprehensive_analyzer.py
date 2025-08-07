#!/usr/bin/env python3
"""
Comprehensive Training Analysis Script for SeqSetVAE
Includes:
- Training curves analysis from TensorBoard logs
- Posterior collapse detection and monitoring
- Performance metrics analysis
- Training dynamics visualization
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import warnings
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Set matplotlib Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosteriorMetricsMonitor(Callback):
    """
    Comprehensive posterior metrics monitoring callback
    
    Monitors four key posterior metrics:
    1. KL divergence
    2. Latent variable variance
    3. Active units ratio
    4. Reconstruction loss
    
    Updates metrics every N steps and saves plots periodically.
    """
    
    def __init__(
        self,
        # Monitoring settings
        update_frequency: int = 50,           # Update metrics every N steps
        plot_frequency: int = 500,            # Save plot every N steps
        window_size: int = 100,               # History window size
        
        # Output settings
        log_dir: str = "./posterior_metrics", # Log save directory
        verbose: bool = True,                 # Whether to output information
    ):
        super().__init__()
        
        # Save parameters
        self.update_frequency = update_frequency
        self.plot_frequency = plot_frequency
        self.window_size = window_size
        self.log_dir = log_dir
        self.verbose = verbose
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize monitoring variables
        self.reset_monitoring_state()
        
    def reset_monitoring_state(self):
        """Reset monitoring state"""
        # Historical data storage (using deque for sliding window)
        self.steps_history = deque(maxlen=self.window_size)
        self.kl_history = deque(maxlen=self.window_size)
        self.var_history = deque(maxlen=self.window_size)
        self.active_units_history = deque(maxlen=self.window_size)
        self.recon_loss_history = deque(maxlen=self.window_size)
        
        # Step counting
        self.global_step = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch"""
        self.global_step += 1
        
        # Only update metrics every N steps
        if self.global_step % self.update_frequency != 0:
            return
            
        # Extract metrics from the model
        metrics = self.extract_metrics(pl_module, outputs)
        if metrics is None:
            return
            
        # Update history
        self.update_history(metrics)
        
        # Save plot periodically
        if self.global_step % self.plot_frequency == 0:
            self.save_metrics_plot()
            
        if self.verbose:
            print(f"📊 Step {self.global_step}: KL={metrics['kl_divergence']:.6f}, "
                  f"Var={metrics['variance']:.6f}, Active={metrics['active_units']:.3f}, "
                  f"Recon={metrics['reconstruction_loss']:.6f}")
    
    def extract_metrics(self, pl_module: LightningModule, outputs) -> Optional[Dict]:
        """Extract posterior metrics from the model"""
        try:
            # Get KL divergence from logged metrics
            kl_divergence = 0.0
            if hasattr(pl_module, 'logged_metrics') and 'train_kl' in pl_module.logged_metrics:
                kl_divergence = pl_module.logged_metrics['train_kl']
            elif hasattr(outputs, 'kl_loss'):
                kl_divergence = outputs.kl_loss.item()
            
            # Extract latent statistics
            latent_stats = self.extract_latent_statistics(pl_module)
            
            # Get reconstruction loss
            reconstruction_loss = 0.0
            if hasattr(pl_module, 'logged_metrics') and 'train_recon_loss' in pl_module.logged_metrics:
                reconstruction_loss = pl_module.logged_metrics['train_recon_loss']
            elif hasattr(outputs, 'recon_loss'):
                reconstruction_loss = outputs.recon_loss.item()
            
            return {
                'kl_divergence': kl_divergence,
                'variance': latent_stats['mean_variance'],
                'active_units': latent_stats['active_units_ratio'],
                'reconstruction_loss': reconstruction_loss
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
            return None
    
    def extract_latent_statistics(self, pl_module: LightningModule) -> Dict:
        """Extract latent variable statistics"""
        try:
            # Get the latest latent representations
            if hasattr(pl_module, 'last_z_means') and hasattr(pl_module, 'last_z_logvars'):
                z_means = pl_module.last_z_means
                z_logvars = pl_module.last_z_logvars
            else:
                # Fallback: try to get from model state
                return {
                    'mean_variance': 0.0,
                    'active_units_ratio': 0.0
                }
            
            # Calculate statistics across all levels
            all_variances = []
            all_active_units = []
            
            for z_mean, z_logvar in zip(z_means, z_logvars):
                if z_mean is not None and z_logvar is not None:
                    # Calculate variance
                    variance = torch.exp(z_logvar)
                    all_variances.append(variance.mean().item())
                    
                    # Calculate active units ratio
                    active_units = (variance > 0.01).float().mean()
                    all_active_units.append(active_units.item())
            
            return {
                'mean_variance': np.mean(all_variances) if all_variances else 0.0,
                'active_units_ratio': np.mean(all_active_units) if all_active_units else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract latent statistics: {e}")
            return {
                'mean_variance': 0.0,
                'active_units_ratio': 0.0
            }
    
    def update_history(self, metrics: Dict):
        """Update historical data"""
        self.steps_history.append(self.global_step)
        self.kl_history.append(metrics['kl_divergence'])
        self.var_history.append(metrics['variance'])
        self.active_units_history.append(metrics['active_units'])
        self.recon_loss_history.append(metrics['reconstruction_loss'])
    
    def save_metrics_plot(self):
        """Save metrics plot"""
        if len(self.steps_history) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Posterior Metrics Monitoring - Step {self.global_step}', fontsize=16)
        
        # KL Divergence
        axes[0, 0].plot(list(self.steps_history), list(self.kl_history), 'b-', linewidth=2)
        axes[0, 0].set_title('KL Divergence')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('KL Divergence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Variance
        axes[0, 1].plot(list(self.steps_history), list(self.var_history), 'orange', linewidth=2)
        axes[0, 1].set_title('Latent Variable Variance')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Mean Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Active Units Ratio
        axes[1, 0].plot(list(self.steps_history), list(self.active_units_history), 'green', linewidth=2)
        axes[1, 0].set_title('Active Units Ratio')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Active Units Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reconstruction Loss
        axes[1, 1].plot(list(self.steps_history), list(self.recon_loss_history), 'purple', linewidth=2)
        axes[1, 1].set_title('Reconstruction Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Reconstruction Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.log_dir, f'posterior_metrics_step_{self.global_step}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        metrics_data = {
            'step': self.global_step,
            'metrics': {
                'kl_divergence': self.kl_history[-1],
                'variance': self.var_history[-1],
                'active_ratio': self.active_units_history[-1],
                'reconstruction_loss': self.recon_loss_history[-1]
            }
        }
        
        json_path = os.path.join(self.log_dir, f'metrics_step_{self.global_step}.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f)
    
    def on_train_end(self, trainer, pl_module):
        """Called at the end of training"""
        # Save final comprehensive analysis
        self.save_final_analysis()
    
    def save_final_analysis(self):
        """Save final comprehensive analysis"""
        if len(self.steps_history) < 2:
            return
            
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Final Training Analysis - Posterior Metrics', fontsize=16, fontweight='bold')
        
        # All metrics on one plot
        ax = axes[0, 0]
        steps = list(self.steps_history)
        ax.plot(steps, list(self.kl_history), 'b-', linewidth=2, label='KL Divergence')
        ax.plot(steps, list(self.var_history), 'orange', linewidth=2, label='Variance')
        ax.plot(steps, list(self.active_units_history), 'green', linewidth=2, label='Active Ratio')
        ax.set_title('All Metrics Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Collapse risk assessment
        ax = axes[0, 1]
        collapse_risk = []
        for kl, var, active in zip(self.kl_history, self.var_history, self.active_units_history):
            risk = 0
            if kl < 0.01:
                risk += 1
            if var < 0.1:
                risk += 1
            if active < 0.1:
                risk += 1
            collapse_risk.append(risk / 3)
        
        ax.plot(steps, collapse_risk, 'red', linewidth=2)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.set_title('Posterior Collapse Risk')
        ax.set_xlabel('Step')
        ax.set_ylabel('Risk Score (0-1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distribution of metrics
        ax = axes[1, 0]
        ax.hist(self.kl_history, bins=30, alpha=0.7, label='KL Divergence', density=True)
        ax.hist(self.var_history, bins=30, alpha=0.7, label='Variance', density=True)
        ax.set_title('Metrics Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Training Summary:
        
        Total Steps: {len(self.steps_history)}
        Final KL Divergence: {self.kl_history[-1]:.6f}
        Final Variance: {self.var_history[-1]:.6f}
        Final Active Ratio: {self.active_units_history[-1]:.3f}
        Final Recon Loss: {self.recon_loss_history[-1]:.6f}
        
        Average KL Divergence: {np.mean(self.kl_history):.6f}
        Average Variance: {np.mean(self.var_history):.6f}
        Average Active Ratio: {np.mean(self.active_units_history):.3f}
        
        Collapse Risk Assessment:
        - Low KL Divergence: {sum(1 for kl in self.kl_history if kl < 0.01)}/{len(self.kl_history)}
        - Low Variance: {sum(1 for var in self.var_history if var < 0.1)}/{len(self.var_history)}
        - Low Active Ratio: {sum(1 for active in self.active_units_history if active < 0.1)}/{len(self.active_units_history)}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # Save final analysis
        final_plot_path = os.path.join(self.log_dir, 'final_analysis.png')
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Final analysis saved to: {final_plot_path}")


def load_tensorboard_logs(log_dir):
    """Load all metrics from TensorBoard event files"""
    print(f"Loading TensorBoard logs from: {log_dir}")
    
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if 'tfevents' in file:
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    print(f"Found {len(event_files)} event files")
    
    # Load events
    all_metrics = {}
    for event_file in event_files:
        print(f"Processing: {event_file}")
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        # Get all scalar tags
        tags = event_acc.Tags()['scalars']
        
        for tag in tags:
            scalar_events = event_acc.Scalars(tag)
            steps = [e.step for e in scalar_events]
            values = [e.value for e in scalar_events]
            
            if tag not in all_metrics:
                all_metrics[tag] = {'steps': [], 'values': []}
            
            all_metrics[tag]['steps'].extend(steps)
            all_metrics[tag]['values'].extend(values)
    
    # Convert to DataFrames
    metrics_df = {}
    for tag, data in all_metrics.items():
        df = pd.DataFrame(data)
        df = df.sort_values('steps').drop_duplicates('steps', keep='last')
        metrics_df[tag] = df
    
    return metrics_df


def plot_loss_curves(metrics_df, save_dir):
    """Plot loss curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Total Loss
    ax = axes[0, 0]
    if 'train_loss_step' in metrics_df:
        ax.plot(metrics_df['train_loss_step']['steps'], 
                metrics_df['train_loss_step']['values'], 
                label='Train Loss', alpha=0.8)
    elif 'train_loss' in metrics_df:
        ax.plot(metrics_df['train_loss']['steps'], 
                metrics_df['train_loss']['values'], 
                label='Train Loss', alpha=0.8)
    if 'val_loss' in metrics_df:
        ax.plot(metrics_df['val_loss']['steps'], 
                metrics_df['val_loss']['values'], 
                label='Val Loss', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KL Loss
    ax = axes[0, 1]
    if 'train_kl_step' in metrics_df:
        ax.plot(metrics_df['train_kl_step']['steps'], 
                metrics_df['train_kl_step']['values'], 
                label='Train KL', alpha=0.8)
    elif 'train_kl' in metrics_df:
        ax.plot(metrics_df['train_kl']['steps'], 
                metrics_df['train_kl']['values'], 
                label='Train KL', alpha=0.8)
    if 'val_kl' in metrics_df:
        ax.plot(metrics_df['val_kl']['steps'], 
                metrics_df['val_kl']['values'], 
                label='Val KL', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('KL Loss')
    ax.set_title('KL Divergence Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Reconstruction Loss
    ax = axes[1, 0]
    if 'train_recon_loss' in metrics_df:
        ax.plot(metrics_df['train_recon_loss']['steps'], 
                metrics_df['train_recon_loss']['values'], 
                label='Train Recon', alpha=0.8)
    if 'val_recon_loss' in metrics_df:
        ax.plot(metrics_df['val_recon_loss']['steps'], 
                metrics_df['val_recon_loss']['values'], 
                label='Val Recon', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Classification Loss
    ax = axes[1, 1]
    if 'train_class_loss' in metrics_df:
        ax.plot(metrics_df['train_class_loss']['steps'], 
                metrics_df['train_class_loss']['values'], 
                label='Train Class', alpha=0.8)
    if 'val_class_loss' in metrics_df:
        ax.plot(metrics_df['val_class_loss']['steps'], 
                metrics_df['val_class_loss']['values'], 
                label='Val Class', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Classification Loss')
    ax.set_title('Classification Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss curves saved to: {save_path}")


def plot_performance_metrics(metrics_df, save_dir):
    """Plot performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Learning Rate
    ax = axes[0, 0]
    if 'lr' in metrics_df:
        ax.plot(metrics_df['lr']['steps'], metrics_df['lr']['values'], 'purple', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    if 'train_acc' in metrics_df:
        ax.plot(metrics_df['train_acc']['steps'], 
                metrics_df['train_acc']['values'], 
                label='Train Acc', alpha=0.8)
    if 'val_acc' in metrics_df:
        ax.plot(metrics_df['val_acc']['steps'], 
                metrics_df['val_acc']['values'], 
                label='Val Acc', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Gradient Norm
    ax = axes[1, 0]
    if 'grad_norm' in metrics_df:
        ax.plot(metrics_df['grad_norm']['steps'], metrics_df['grad_norm']['values'], 'red', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)
    
    # 4. Beta Value (if using KL annealing)
    ax = axes[1, 1]
    if 'beta' in metrics_df:
        ax.plot(metrics_df['beta']['steps'], metrics_df['beta']['values'], 'green', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Beta')
        ax.set_title('KL Annealing Beta')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'performance_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Performance metrics saved to: {save_path}")


def plot_training_dynamics(metrics_df, save_dir):
    """Plot training dynamics analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Loss components ratio
    ax = axes[0, 0]
    if all(key in metrics_df for key in ['train_kl', 'train_recon_loss']):
        kl_values = metrics_df['train_kl']['values']
        recon_values = metrics_df['train_recon_loss']['values']
        steps = metrics_df['train_kl']['steps']
        
        # Calculate ratio
        ratio = np.array(kl_values) / (np.array(kl_values) + np.array(recon_values) + 1e-8)
        ax.plot(steps, ratio, 'blue', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('KL / (KL + Recon)')
        ax.set_title('KL vs Reconstruction Loss Ratio')
        ax.grid(True, alpha=0.3)
    
    # 2. Loss correlation
    ax = axes[0, 1]
    if all(key in metrics_df for key in ['train_kl', 'train_recon_loss']):
        kl_values = metrics_df['train_kl']['values']
        recon_values = metrics_df['train_recon_loss']['values']
        ax.scatter(kl_values, recon_values, alpha=0.6)
        ax.set_xlabel('KL Loss')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('KL vs Reconstruction Loss Correlation')
        ax.grid(True, alpha=0.3)
    
    # 3. Training stability
    ax = axes[1, 0]
    if 'train_loss' in metrics_df:
        loss_values = metrics_df['train_loss']['values']
        steps = metrics_df['train_loss']['steps']
        
        # Calculate moving average
        window = min(50, len(loss_values) // 10)
        if window > 1:
            moving_avg = pd.Series(loss_values).rolling(window=window).mean()
            ax.plot(steps, loss_values, alpha=0.5, label='Raw Loss')
            ax.plot(steps, moving_avg, 'red', linewidth=2, label=f'Moving Avg (w={window})')
            ax.legend()
        else:
            ax.plot(steps, loss_values, 'blue', linewidth=2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Stability')
        ax.grid(True, alpha=0.3)
    
    # 4. Convergence analysis
    ax = axes[1, 1]
    if 'train_loss' in metrics_df:
        loss_values = metrics_df['train_loss']['values']
        steps = metrics_df['train_loss']['steps']
        
        # Calculate relative improvement
        if len(loss_values) > 10:
            baseline = np.mean(loss_values[:10])
            relative_improvement = (baseline - np.array(loss_values)) / baseline
            ax.plot(steps, relative_improvement, 'green', linewidth=2)
            ax.set_xlabel('Steps')
            ax.set_ylabel('Relative Improvement')
            ax.set_title('Convergence Analysis')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_dynamics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training dynamics saved to: {save_path}")


def analyze_collapse_indicators(metrics_df, save_dir):
    """Analyze posterior collapse indicators from TensorBoard logs"""
    print("Analyzing posterior collapse indicators...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. KL divergence analysis
    ax = axes[0, 0]
    kl_metrics = [key for key in metrics_df.keys() if 'kl' in key.lower()]
    
    for metric in kl_metrics:
        if 'train' in metric:
            ax.plot(metrics_df[metric]['steps'], 
                   metrics_df[metric]['values'], 
                   label=metric, alpha=0.8)
    
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
    ax.set_xlabel('Steps')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Variance analysis (if available)
    ax = axes[0, 1]
    var_metrics = [key for key in metrics_df.keys() if 'var' in key.lower()]
    
    for metric in var_metrics:
        ax.plot(metrics_df[metric]['steps'], 
               metrics_df[metric]['values'], 
               label=metric, alpha=0.8)
    
    if var_metrics:
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Variance')
        ax.set_title('Latent Variable Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No variance metrics found', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Variance Analysis (No Data)')
    
    # 3. Active units analysis (if available)
    ax = axes[1, 0]
    active_metrics = [key for key in metrics_df.keys() if 'active' in key.lower()]
    
    for metric in active_metrics:
        ax.plot(metrics_df[metric]['steps'], 
               metrics_df[metric]['values'], 
               label=metric, alpha=0.8)
    
    if active_metrics:
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Active Units Ratio')
        ax.set_title('Active Units Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No active units metrics found', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Active Units Analysis (No Data)')
    
    # 4. Collapse risk summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate collapse risk
    collapse_indicators = []
    
    for metric in kl_metrics:
        if 'train' in metric:
            values = metrics_df[metric]['values']
            low_kl_count = sum(1 for v in values if v < 0.01)
            collapse_indicators.append(f"{metric}: {low_kl_count}/{len(values)} steps with low KL")
    
    for metric in var_metrics:
        values = metrics_df[metric]['values']
        low_var_count = sum(1 for v in values if v < 0.1)
        collapse_indicators.append(f"{metric}: {low_var_count}/{len(values)} steps with low variance")
    
    for metric in active_metrics:
        values = metrics_df[metric]['values']
        low_active_count = sum(1 for v in values if v < 0.1)
        collapse_indicators.append(f"{metric}: {low_active_count}/{len(values)} steps with low active ratio")
    
    summary_text = "Posterior Collapse Risk Assessment:\n\n"
    if collapse_indicators:
        summary_text += "\n".join(collapse_indicators)
    else:
        summary_text += "No collapse indicators found in logs"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'collapse_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Collapse analysis saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Training Analysis')
    parser.add_argument('--mode', type=str, choices=['tensorboard', 'callback'], default='tensorboard',
                       help='Analysis mode: tensorboard logs or callback monitoring')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing TensorBoard logs or monitoring data')
    parser.add_argument('--save_dir', type=str, default='./analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--update_frequency', type=int, default=50,
                       help='Update frequency for callback monitoring (steps)')
    parser.add_argument('--plot_frequency', type=int, default=500,
                       help='Plot frequency for callback monitoring (steps)')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == 'tensorboard':
        # TensorBoard analysis mode
        print("🔍 Running TensorBoard analysis mode...")
        
        try:
            # Load TensorBoard logs
            metrics_df = load_tensorboard_logs(args.log_dir)
            
            # Create comprehensive analysis
            plot_loss_curves(metrics_df, args.save_dir)
            plot_performance_metrics(metrics_df, args.save_dir)
            plot_training_dynamics(metrics_df, args.save_dir)
            analyze_collapse_indicators(metrics_df, args.save_dir)
            
            print(f"✅ Analysis completed! Results saved to: {args.save_dir}")
            
        except Exception as e:
            print(f"❌ Error during TensorBoard analysis: {e}")
            print("Make sure TensorBoard logs exist in the specified directory")
    
    elif args.mode == 'callback':
        # Callback monitoring mode
        print("🔍 Running callback monitoring mode...")
        print("This mode is intended to be used as a callback during training.")
        print("To use this as a standalone tool, use 'tensorboard' mode instead.")
        
        # Create callback instance for reference
        monitor = PosteriorMetricsMonitor(
            update_frequency=args.update_frequency,
            plot_frequency=args.plot_frequency,
            log_dir=args.log_dir
        )
        
        print(f"✅ Callback created with log directory: {args.log_dir}")
        print("Add this callback to your trainer callbacks list during training.")


if __name__ == "__main__":
    main()