import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
import logging
from typing import Optional, Dict, Any
import json
from datetime import datetime
import lightning.pytorch as pl

class PosteriorMetricsMonitor(pl.Callback):
    """
    Monitors and detects posterior collapse during VAE training.
    
    This class tracks KL divergence and reconstruction loss to identify
    potential posterior collapse scenarios and provides real-time monitoring
    with adaptive thresholds.
    """
    
    def __init__(self, 
                 update_frequency: int = 50,
                 plot_frequency: int = 500,
                 window_size: int = 100,
                 log_dir: str = "posterior_metrics",
                 verbose: bool = True,
                 kl_threshold: float = 0.1,
                 reconstruction_threshold: float = 0.01):
        """
        Initialize the posterior metrics monitor.
        
        Args:
            update_frequency: How often to update metrics (in steps)
            plot_frequency: How often to generate plots (in steps)
            window_size: Size of the sliding window for trend analysis
            log_dir: Directory to save monitoring data and plots
            verbose: Whether to print monitoring information
            kl_threshold: Threshold for KL divergence collapse detection
            reconstruction_threshold: Threshold for reconstruction loss collapse detection
        """
        self.update_frequency = update_frequency
        self.plot_frequency = plot_frequency
        self.window_size = window_size
        self.log_dir = log_dir
        self.verbose = verbose
        self.kl_threshold = kl_threshold
        self.reconstruction_threshold = reconstruction_threshold
        
        # Initialize storage for metrics
        self.kl_divergences = deque(maxlen=window_size)
        self.reconstruction_losses = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)
        
        # Collapse detection state
        self.kl_collapse_detected = False
        self.reconstruction_collapse_detected = False
        self.collapse_warnings = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize plot
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Posterior Collapse Monitoring', fontsize=14)
        
        self.step_count = 0
        
    def update(self, step: int, kl_divergence: float, reconstruction_loss: float, 
               additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Update the monitoring metrics.
        
        Args:
            step: Current training step
            kl_divergence: KL divergence value
            reconstruction_loss: Reconstruction loss value
            additional_metrics: Additional metrics to track
        """
        self.step_count += 1
        
        # Only update at specified frequency
        if self.step_count % self.update_frequency != 0:
            return
        
        # Store metrics
        self.kl_divergences.append(kl_divergence)
        self.reconstruction_losses.append(reconstruction_loss)
        self.steps.append(step)
        
        # Check for posterior collapse
        self._check_collapse(step, kl_divergence, reconstruction_loss)
        
        # Generate plots at specified frequency
        if self.step_count % self.plot_frequency == 0:
            self._generate_plots()
            
        # Save metrics to file
        self._save_metrics(step, kl_divergence, reconstruction_loss, additional_metrics)
        
    def _check_collapse(self, step: int, kl_divergence: float, reconstruction_loss: float):
        """
        Check for signs of posterior collapse.
        
        Args:
            step: Current training step
            kl_divergence: KL divergence value
            reconstruction_loss: Reconstruction loss value
        """
        if len(self.kl_divergences) < 10:  # Need minimum data points
            return
            
        # Check KL divergence collapse
        recent_kl = list(self.kl_divergences)[-10:]
        kl_trend = np.mean(recent_kl[-5:]) - np.mean(recent_kl[:5])
        
        if kl_divergence < self.kl_threshold and kl_trend < -0.01:
            if not self.kl_collapse_detected:
                self.kl_collapse_detected = True
                warning_msg = f"⚠️  KL divergence collapse detected at step {step}: {kl_divergence:.6f}"
                self.collapse_warnings.append({
                    'step': step,
                    'type': 'kl_collapse',
                    'value': kl_divergence,
                    'message': warning_msg
                })
                if self.verbose:
                    print(warning_msg)
        
        # Check reconstruction loss collapse
        recent_recon = list(self.reconstruction_losses)[-10:]
        recon_trend = np.mean(recent_recon[-5:]) - np.mean(recent_recon[:5])
        
        if reconstruction_loss < self.reconstruction_threshold and recon_trend < -0.001:
            if not self.reconstruction_collapse_detected:
                self.reconstruction_collapse_detected = True
                warning_msg = f"⚠️  Reconstruction loss collapse detected at step {step}: {reconstruction_loss:.6f}"
                self.collapse_warnings.append({
                    'step': step,
                    'type': 'reconstruction_collapse',
                    'value': reconstruction_loss,
                    'message': warning_msg
                })
                if self.verbose:
                    print(warning_msg)
    
    def _generate_plots(self):
        """Generate real-time monitoring plots."""
        if len(self.steps) < 2:
            return
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        steps_list = list(self.steps)
        kl_list = list(self.kl_divergences)
        recon_list = list(self.reconstruction_losses)
        
        # Plot KL divergence
        self.ax1.plot(steps_list, kl_list, 'b-', label='KL Divergence')
        self.ax1.axhline(y=self.kl_threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({self.kl_threshold})')
        self.ax1.set_ylabel('KL Divergence')
        self.ax1.set_title('KL Divergence Monitoring')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot reconstruction loss
        self.ax2.plot(steps_list, recon_list, 'g-', label='Reconstruction Loss')
        self.ax2.axhline(y=self.reconstruction_threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({self.reconstruction_threshold})')
        self.ax2.set_xlabel('Training Step')
        self.ax2.set_ylabel('Reconstruction Loss')
        self.ax2.set_title('Reconstruction Loss Monitoring')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, f'posterior_monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if self.verbose:
            print(f"📊 Monitoring plot saved: {plot_path}")
    
    def _save_metrics(self, step: int, kl_divergence: float, reconstruction_loss: float, 
                     additional_metrics: Optional[Dict[str, Any]] = None):
        """Save metrics to JSON file."""
        metrics_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'kl_divergence': float(kl_divergence),
            'reconstruction_loss': float(reconstruction_loss),
            'kl_collapse_detected': self.kl_collapse_detected,
            'reconstruction_collapse_detected': self.reconstruction_collapse_detected
        }
        
        # Prepare nested metrics structure for realtime visualizer
        nested_metrics = {
            'step': step,
            'metrics': {
                'kl_divergence': float(kl_divergence),
                'reconstruction_loss': float(reconstruction_loss),
            }
        }
        
        if additional_metrics:
            metrics_data.update(additional_metrics)
            # add into nested structure too
            for k, v in additional_metrics.items():
                nested_metrics['metrics'][k] = v
        
        # Save to history file (JSON array)
        metrics_file = os.path.join(self.log_dir, 'metrics_history.json')
        # Load existing data or create new
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        else:
            history = []
        history.append(metrics_data)
        # Keep recent history
        if len(history) > 1000:
            history = history[-1000:]
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Additionally, write per-step JSON for realtime visualizer compatibility
        step_file = os.path.join(self.log_dir, f'metrics_step_{step}.json')
        try:
            with open(step_file, 'w') as f:
                json.dump(nested_metrics, f)
        except Exception:
            pass
    
    def get_collapse_status(self) -> Dict[str, Any]:
        """
        Get current collapse detection status.
        
        Returns:
            Dictionary containing collapse status and warnings
        """
        return {
            'kl_collapse_detected': self.kl_collapse_detected,
            'reconstruction_collapse_detected': self.reconstruction_collapse_detected,
            'warnings': self.collapse_warnings.copy(),
            'current_kl': float(self.kl_divergences[-1]) if self.kl_divergences else None,
            'current_reconstruction': float(self.reconstruction_losses[-1]) if self.reconstruction_losses else None
        }
    
    def reset(self):
        """Reset the monitor state."""
        self.kl_divergences.clear()
        self.reconstruction_losses.clear()
        self.steps.clear()
        self.kl_collapse_detected = False
        self.reconstruction_collapse_detected = False
        self.collapse_warnings.clear()
        self.step_count = 0
        
        if self.verbose:
            print("🔄 Posterior metrics monitor reset")
    
    def close(self):
        """Close the monitor and cleanup resources."""
        plt.close(self.fig)
        if self.verbose:
            print("📊 Posterior metrics monitor closed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass
    
    # PyTorch Lightning callback methods
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        # Prefer reading from pl_module.logged_metrics since outputs may be a scalar
        kl_loss = None
        recon_loss = None
        if hasattr(pl_module, 'logged_metrics') and isinstance(pl_module.logged_metrics, dict):
            kl_loss = pl_module.logged_metrics.get('train_kl', None)
            recon_loss = pl_module.logged_metrics.get('train_recon', None)
            if torch.is_tensor(kl_loss):
                kl_loss = kl_loss.item()
            if torch.is_tensor(recon_loss):
                recon_loss = recon_loss.item()
        
        # Fallback to outputs dict if available
        if (kl_loss is None or recon_loss is None) and outputs is not None and isinstance(outputs, dict):
            kl_loss = outputs.get('kl_loss', kl_loss)
            recon_loss = outputs.get('recon_loss', recon_loss)
            if torch.is_tensor(kl_loss):
                kl_loss = kl_loss.item()
            if torch.is_tensor(recon_loss):
                recon_loss = recon_loss.item()
        
        # Default to zeros if still missing
        kl_loss = float(kl_loss) if kl_loss is not None else 0.0
        recon_loss = float(recon_loss) if recon_loss is not None else 0.0
        
        # Compute additional latent statistics from the model if available
        variance_value = None
        active_ratio_value = None
        try:
            z_list = None
            if hasattr(pl_module, '_last_z_list') and pl_module._last_z_list:
                z_list = pl_module._last_z_list
            elif hasattr(pl_module, 'setvae') and hasattr(pl_module.setvae, '_last_z_list'):
                z_list = pl_module.setvae._last_z_list
            if z_list:
                variances = []
                actives = []
                for z_sample, mu, logvar in z_list:
                    var = torch.exp(logvar)
                    variances.append(var.mean().item())
                    actives.append((var > 0.01).float().mean().item())
                if variances:
                    variance_value = float(np.mean(variances))
                if actives:
                    active_ratio_value = float(np.mean(actives))
        except Exception:
            pass
        
        additional = {}
        if variance_value is not None:
            additional['variance'] = variance_value
        if active_ratio_value is not None:
            additional['active_ratio'] = active_ratio_value
        
        # Update monitoring and save
        self.update(trainer.global_step, kl_loss, recon_loss, additional)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        # Generate plots at epoch end
        if self.step_count % self.plot_frequency == 0:
            self._generate_plots()
    
    def on_fit_end(self, trainer, pl_module):
        """Called when training ends."""
        # Generate final plots and cleanup
        self._generate_plots()
        self.close()