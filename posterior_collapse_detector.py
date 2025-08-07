import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule
import warnings
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PosteriorMetricsMonitor(Callback):
    """
    Simple posterior metrics monitoring callback
    
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
                kl_divergence = pl_module.logged_metrics['train_kl'].item()
            elif hasattr(outputs, 'kl_loss'):
                kl_divergence = outputs['kl_loss'].item()
            
            # Get reconstruction loss
            recon_loss = 0.0
            if hasattr(pl_module, 'logged_metrics') and 'train_recon' in pl_module.logged_metrics:
                recon_loss = pl_module.logged_metrics['train_recon'].item()
            elif hasattr(outputs, 'recon_loss'):
                recon_loss = outputs['recon_loss'].item()
            
            # Extract latent variable statistics
            latent_stats = self.extract_latent_statistics(pl_module)
            
            return {
                'kl_divergence': kl_divergence,
                'variance': latent_stats['variance'],
                'active_units': latent_stats['active_units'],
                'reconstruction_loss': recon_loss,
            }
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to extract metrics: {e}")
            return None
    
    def extract_latent_statistics(self, pl_module: LightningModule) -> Dict:
        """Extract latent variable statistics from the model"""
        try:
            # Try to get latent variables from the model
            if hasattr(pl_module, '_last_z_list') and pl_module._last_z_list is not None:
                # Extract from _last_z_list
                z_list = pl_module._last_z_list
                if isinstance(z_list, list) and len(z_list) > 0:
                    # Concatenate all latent variables
                    z_tensor = torch.cat([z for z in z_list if z is not None], dim=0)
                else:
                    z_tensor = z_list
            elif hasattr(pl_module, 'setvae') and hasattr(pl_module.setvae, '_last_z_list'):
                # Try to get from setvae
                z_list = pl_module.setvae._last_z_list
                if isinstance(z_list, list) and len(z_list) > 0:
                    z_tensor = torch.cat([z for z in z_list if z is not None], dim=0)
                else:
                    z_tensor = z_list
            else:
                # Fallback: return default values
                return {
                    'variance': 0.0,
                    'active_units': 0.0,
                }
            
            if z_tensor is None or z_tensor.numel() == 0:
                return {
                    'variance': 0.0,
                    'active_units': 0.0,
                }
            
            # Calculate variance (assuming z_tensor is [batch_size, latent_dim])
            if z_tensor.dim() == 2:
                variance = z_tensor.var(dim=0).mean().item()
            else:
                variance = z_tensor.var().item()
            
            # Calculate active units ratio (variance > 0.01)
            if z_tensor.dim() == 2:
                active_units = (z_tensor.var(dim=0) > 0.01).float().mean().item()
            else:
                active_units = 0.0
            
            return {
                'variance': variance,
                'active_units': active_units,
            }
            
        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to extract latent statistics: {e}")
            return {
                'variance': 0.0,
                'active_units': 0.0,
            }
    
    def update_history(self, metrics: Dict):
        """Update history with new metrics"""
        self.steps_history.append(self.global_step)
        self.kl_history.append(metrics['kl_divergence'])
        self.var_history.append(metrics['variance'])
        self.active_units_history.append(metrics['active_units'])
        self.recon_loss_history.append(metrics['reconstruction_loss'])
    
    def save_metrics_plot(self):
        """Save metrics plot"""
        try:
            if len(self.steps_history) == 0:
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Posterior Metrics Monitoring - Step {self.global_step}', fontsize=16)
            
            steps = list(self.steps_history)
            
            # KL divergence history
            if self.kl_history:
                axes[0, 0].plot(steps, list(self.kl_history), 'b-', linewidth=2)
                axes[0, 0].set_title('KL Divergence')
                axes[0, 0].set_ylabel('KL Divergence')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_yscale('log')
                
            # Variance history
            if self.var_history:
                axes[0, 1].plot(steps, list(self.var_history), 'orange', linewidth=2)
                axes[0, 1].set_title('Latent Variable Variance')
                axes[0, 1].set_ylabel('Mean Variance')
                axes[0, 1].grid(True, alpha=0.3)
                
            # Active units ratio history
            if self.active_units_history:
                axes[1, 0].plot(steps, list(self.active_units_history), 'green', linewidth=2)
                axes[1, 0].set_title('Active Units Ratio')
                axes[1, 0].set_ylabel('Active Units Ratio')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_ylim([0, 1])
                
            # Reconstruction loss history
            if self.recon_loss_history:
                axes[1, 1].plot(steps, list(self.recon_loss_history), 'purple', linewidth=2)
                axes[1, 1].set_title('Reconstruction Loss')
                axes[1, 1].set_ylabel('Reconstruction Loss')
                axes[1, 1].grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.log_dir, f"posterior_metrics_step_{self.global_step}_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f"📊 Posterior metrics plot saved: {plot_path}")
                
        except Exception as e:
            logger.warning(f"Failed to save metrics plot: {e}")
            
    def on_train_end(self, trainer, pl_module):
        """Save final plot at the end of training"""
        if len(self.steps_history) > 0:
            self.save_metrics_plot()
            
        summary_msg = f"""
        
📊 Posterior Metrics Monitoring Summary:
===========================================
Total steps monitored: {len(self.steps_history)}
Update frequency: {self.update_frequency}
Plot frequency: {self.plot_frequency}
Log directory: {self.log_dir}
        """
        
        logger.info(summary_msg)
        if self.verbose:
            print(summary_msg)