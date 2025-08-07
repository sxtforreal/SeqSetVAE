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

class PosteriorCollapseDetector(Callback):
    """
    Real-time VAE posterior collapse detection callback class
    
    Monitors multiple key indicators:
    1. KL divergence (per layer and overall)
    2. Latent variable variance
    3. Latent variable mean distribution
    4. Number of active units
    5. Reconstruction error changes
    """
    
    def __init__(
        self,
        # Detection thresholds
        kl_threshold: float = 0.01,          # KL divergence low threshold
        var_threshold: float = 0.1,          # Variance low threshold
        active_units_threshold: float = 0.1,  # Active units ratio threshold
        
        # Monitoring window
        window_size: int = 100,              # Sliding window size
        check_frequency: int = 50,           # Check frequency (check every N steps)
        
        # Early stopping
        early_stop_patience: int = 200,      # How many consecutive steps of collapse detection before stopping
        auto_save_on_collapse: bool = True,  # Auto save when collapse is detected
        
        # Output settings
        log_dir: str = "./collapse_logs",    # Log save directory
        plot_frequency: int = 500,           # Plot frequency
        verbose: bool = True,                # Whether to output detailed information
    ):
        super().__init__()
        
        # Save parameters
        self.kl_threshold = kl_threshold
        self.var_threshold = var_threshold
        self.active_units_threshold = active_units_threshold
        
        self.window_size = window_size
        self.check_frequency = check_frequency
        
        self.early_stop_patience = early_stop_patience
        self.auto_save_on_collapse = auto_save_on_collapse
        
        self.log_dir = log_dir
        self.plot_frequency = plot_frequency
        self.verbose = verbose
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize monitoring variables
        self.reset_monitoring_state()
        
        # Set up file logging
        self.setup_file_logging()
        
    def reset_monitoring_state(self):
        """Reset monitoring state"""
        # Historical data storage (using deque for sliding window)
        self.kl_history = deque(maxlen=self.window_size)
        self.var_history = deque(maxlen=self.window_size)
        self.mean_history = deque(maxlen=self.window_size)
        self.active_units_history = deque(maxlen=self.window_size)
        self.recon_loss_history = deque(maxlen=self.window_size)
        
        # Collapse detection state
        self.collapse_detected = False
        self.collapse_step = None
        self.collapse_consecutive_steps = 0
        self.collapse_warnings = []
        
        # Step counting
        self.global_step = 0
        
        # Statistics
        self.collapse_stats = {
            'total_checks': 0,
            'warnings_issued': 0,
            'false_alarms': 0,
        }
        
    def setup_file_logging(self):
        """Set up file logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"collapse_detection_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        self.log_file = log_file
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Perform detection after each training batch ends"""
        self.global_step += 1
        
        # Check at specified frequency
        if self.global_step % self.check_frequency != 0:
            return
            
        # Extract monitoring metrics
        metrics = self.extract_monitoring_metrics(pl_module, outputs)
        if metrics is None:
            return
            
        # Update historical data
        self.update_history(metrics)
        
        # Log all metrics periodically
        if self.global_step % self.check_frequency == 0:
            metric_log = f"Step {self.global_step} - Metrics: "
            metric_parts = []
            
            if 'kl_divergence' in metrics:
                metric_parts.append(f"KL: {metrics['kl_divergence']:.6f}")
            if 'mean_variance' in metrics:
                metric_parts.append(f"Variance: {metrics['mean_variance']:.6f}")
            if 'mean_active_ratio' in metrics:
                metric_parts.append(f"Active Units: {metrics['mean_active_ratio']:.3f}")
            if 'recon_loss' in metrics:
                metric_parts.append(f"Recon Loss: {metrics['recon_loss']:.6f}")
            
            if metric_parts:
                metric_log += ", ".join(metric_parts)
                logger.info(metric_log)
                
            # Log detailed layer-wise metrics if available
            if 'layer_variances' in metrics and self.verbose:
                for i, var in enumerate(metrics['layer_variances']):
                    layer_log = f"  Layer {i}: Variance={var:.6f}"
                    if 'active_units_ratios' in metrics and i < len(metrics['active_units_ratios']):
                        layer_log += f", Active Ratio={metrics['active_units_ratios'][i]:.3f}"
                    if 'layer_mean_magnitudes' in metrics and i < len(metrics['layer_mean_magnitudes']):
                        layer_log += f", Mean Magnitude={metrics['layer_mean_magnitudes'][i]:.6f}"
                    logger.info(layer_log)
        
        # Perform collapse detection
        collapse_detected, warnings = self.detect_collapse(metrics)
        
        # Handle detection results
        self.handle_detection_results(trainer, pl_module, collapse_detected, warnings)
        
        # Periodic plotting and saving
        if self.global_step % self.plot_frequency == 0:
            self.save_monitoring_plots()
            
    def extract_monitoring_metrics(self, pl_module: LightningModule, outputs) -> Optional[Dict]:
        """Extract monitoring metrics from model"""
        try:
            metrics = {}
            
            # 1. Extract KL divergence (from logged metrics)
            if hasattr(pl_module, 'logged_metrics'):
                logged = pl_module.logged_metrics
                if 'train_kl' in logged:
                    metrics['kl_divergence'] = logged['train_kl'].item()
                if 'train_recon' in logged:
                    metrics['recon_loss'] = logged['train_recon'].item()
                    
            # 2. Extract latent variable statistics directly from model
            if hasattr(pl_module, 'setvae'):
                # Get latent variables from the most recent forward pass
                latent_stats = self.extract_latent_statistics(pl_module)
                if latent_stats:
                    metrics.update(latent_stats)
                    
            return metrics if metrics else None
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
            return None
            
    def extract_latent_statistics(self, pl_module: LightningModule) -> Dict:
        """Extract latent variable statistics"""
        stats = {}
        
        try:
            # Run a small batch to get latent variable statistics
            pl_module.eval()
            with torch.no_grad():
                # This needs to be adjusted based on your specific model structure
                # Assume we can get recent latent variables through some method
                
                # If model has stored recent latent variables
                if hasattr(pl_module, '_last_z_list') and pl_module._last_z_list:
                    z_list = pl_module._last_z_list
                    
                    # Calculate statistics for each layer
                    layer_vars = []
                    layer_means = []
                    active_units_ratios = []
                    
                    for layer_idx, (z_sample, mu, logvar) in enumerate(z_list):
                        # Variance statistics
                        var = torch.exp(logvar)
                        mean_var = var.mean().item()
                        layer_vars.append(mean_var)
                        
                        # Mean statistics
                        mean_abs_mu = torch.abs(mu).mean().item()
                        layer_means.append(mean_abs_mu)
                        
                        # Active unit statistics (ratio of units with variance above threshold)
                        active_ratio = (var.mean(0) > self.var_threshold).float().mean().item()
                        active_units_ratios.append(active_ratio)
                        
                    stats['layer_variances'] = layer_vars
                    stats['layer_mean_magnitudes'] = layer_means
                    stats['active_units_ratios'] = active_units_ratios
                    
                    # Overall statistics
                    stats['mean_variance'] = np.mean(layer_vars)
                    stats['mean_active_ratio'] = np.mean(active_units_ratios)
                    
            pl_module.train()
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to extract latent statistics: {e}")
            return {}
            
    def update_history(self, metrics: Dict):
        """Update historical data"""
        if 'kl_divergence' in metrics:
            self.kl_history.append(metrics['kl_divergence'])
            
        if 'mean_variance' in metrics:
            self.var_history.append(metrics['mean_variance'])
            
        if 'mean_active_ratio' in metrics:
            self.active_units_history.append(metrics['mean_active_ratio'])
            
        if 'recon_loss' in metrics:
            self.recon_loss_history.append(metrics['recon_loss'])
            
        if 'layer_mean_magnitudes' in metrics:
            avg_mean_mag = np.mean(metrics['layer_mean_magnitudes'])
            self.mean_history.append(avg_mean_mag)
            
    def detect_collapse(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """Detect posterior collapse"""
        warnings = []
        collapse_indicators = 0
        
        self.collapse_stats['total_checks'] += 1
        
        # 1. KL divergence detection
        if 'kl_divergence' in metrics:
            kl_val = metrics['kl_divergence']
            if kl_val < self.kl_threshold:
                warnings.append(f"KL divergence too low: {kl_val:.6f} < {self.kl_threshold}")
                collapse_indicators += 1
                
            # Check KL divergence trend
            if len(self.kl_history) >= 20:
                recent_kl = list(self.kl_history)[-20:]
                if all(kl < self.kl_threshold for kl in recent_kl):
                    warnings.append("KL divergence consistently low (last 20 steps)")
                    collapse_indicators += 2
                    
        # 2. Variance detection
        if 'mean_variance' in metrics:
            var_val = metrics['mean_variance']
            if var_val < self.var_threshold:
                warnings.append(f"Latent variable variance too low: {var_val:.6f} < {self.var_threshold}")
                collapse_indicators += 1
                
        # 3. Active units detection
        if 'mean_active_ratio' in metrics:
            active_ratio = metrics['mean_active_ratio']
            if active_ratio < self.active_units_threshold:
                warnings.append(f"Active units ratio too low: {active_ratio:.3f} < {self.active_units_threshold}")
                collapse_indicators += 1
                
        # 4. Reconstruction loss trend detection
        if len(self.recon_loss_history) >= 50:
            recent_recon = list(self.recon_loss_history)[-50:]
            recon_trend = np.polyfit(range(len(recent_recon)), recent_recon, 1)[0]
            if recon_trend > 0.001:  # Reconstruction loss continuously rising
                warnings.append(f"Reconstruction loss continuously rising, trend: {recon_trend:.6f}")
                collapse_indicators += 1
                
        # Overall judgment
        collapse_detected = collapse_indicators >= 2  # At least 2 abnormal indicators to determine collapse
        
        if warnings:
            self.collapse_stats['warnings_issued'] += 1
            
        return collapse_detected, warnings
        
    def handle_detection_results(self, trainer, pl_module, collapse_detected: bool, warnings: List[str]):
        """Handle detection results"""
        
        if warnings and self.verbose:
            warning_msg = f"Step {self.global_step}: Posterior collapse warning!\n" + "\n".join(f"  - {w}" for w in warnings)
            logger.warning(warning_msg)
            print(f"\n⚠️  {warning_msg}")
            
        if collapse_detected:
            self.collapse_consecutive_steps += 1
            
            if not self.collapse_detected:
                self.collapse_detected = True
                self.collapse_step = self.global_step
                
                collapse_msg = f"🚨 Posterior collapse detected! Step: {self.global_step}"
                logger.error(collapse_msg)
                print(f"\n{collapse_msg}")
                
                # Auto save model
                if self.auto_save_on_collapse:
                    self.save_model_on_collapse(trainer, pl_module)
                    
            # Check if early stopping is needed
            if self.collapse_consecutive_steps >= self.early_stop_patience:
                stop_msg = f"Posterior collapse detected for {self.early_stop_patience} consecutive steps, recommend stopping training!"
                logger.error(stop_msg)
                print(f"\n🛑 {stop_msg}")
                
                # Here we could set trainer.should_stop = True to stop training
                # But for safety, we only log the recommendation
                
        else:
            # Reset consecutive collapse count
            if self.collapse_consecutive_steps > 0:
                self.collapse_consecutive_steps = 0
                
                if self.collapse_detected:
                    recovery_msg = f"Posterior collapse state recovered, Step: {self.global_step}"
                    logger.info(recovery_msg)
                    print(f"\n✅ {recovery_msg}")
                    
    def save_model_on_collapse(self, trainer, pl_module):
        """Save model when collapse is detected"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.log_dir, f"model_before_collapse_step_{self.global_step}_{timestamp}.ckpt")
            
            trainer.save_checkpoint(save_path)
            logger.info(f"Model saved to: {save_path}")
            print(f"💾 Model saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
    def save_monitoring_plots(self):
        """Save monitoring plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Posterior Collapse Monitoring - Step {self.global_step}', fontsize=16)
            
            # KL divergence history
            if self.kl_history:
                axes[0, 0].plot(list(self.kl_history))
                axes[0, 0].axhline(y=self.kl_threshold, color='r', linestyle='--', alpha=0.7)
                axes[0, 0].set_title('KL Divergence History')
                axes[0, 0].set_ylabel('KL Divergence')
                axes[0, 0].grid(True, alpha=0.3)
                
            # Variance history
            if self.var_history:
                axes[0, 1].plot(list(self.var_history), color='orange')
                axes[0, 1].axhline(y=self.var_threshold, color='r', linestyle='--', alpha=0.7)
                axes[0, 1].set_title('Latent Variable Variance History')
                axes[0, 1].set_ylabel('Mean Variance')
                axes[0, 1].grid(True, alpha=0.3)
                
            # Active units ratio history
            if self.active_units_history:
                axes[1, 0].plot(list(self.active_units_history), color='green')
                axes[1, 0].axhline(y=self.active_units_threshold, color='r', linestyle='--', alpha=0.7)
                axes[1, 0].set_title('Active Units Ratio History')
                axes[1, 0].set_ylabel('Active Units Ratio')
                axes[1, 0].grid(True, alpha=0.3)
                
            # Reconstruction loss history
            if self.recon_loss_history:
                axes[1, 1].plot(list(self.recon_loss_history), color='purple')
                axes[1, 1].set_title('Reconstruction Loss History')
                axes[1, 1].set_ylabel('Reconstruction Loss')
                axes[1, 1].grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.log_dir, f"monitoring_plot_step_{self.global_step}_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f"📊 Monitoring plot saved: {plot_path}")
                
        except Exception as e:
            logger.warning(f"Failed to save monitoring plot: {e}")
            
    def on_train_end(self, trainer, pl_module):
        """Summary at the end of training"""
        summary_msg = f"""
        
🎯 Posterior Collapse Detection Summary:
===========================================
Total checks: {self.collapse_stats['total_checks']}
Warnings issued: {self.collapse_stats['warnings_issued']}
Collapse detected: {'Yes' if self.collapse_detected else 'No'}
Collapse occurrence step: {self.collapse_step if self.collapse_step else 'N/A'}
Log file: {self.log_file}
        """
        
        logger.info(summary_msg)
        print(summary_msg)
        
        # Save final statistics
        stats_file = os.path.join(self.log_dir, "final_statistics.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(summary_msg)
            f.write(f"\nDetection parameters:\n")
            f.write(f"KL threshold: {self.kl_threshold}\n")
            f.write(f"Variance threshold: {self.var_threshold}\n") 
            f.write(f"Active units threshold: {self.active_units_threshold}\n")
            f.write(f"Check frequency: {self.check_frequency}\n")
            f.write(f"Early stop patience: {self.early_stop_patience}\n")


# Helper functions: Add latent variable tracking to existing models
def add_latent_tracking_to_model(model):
    """Add latent variable tracking functionality to model"""
    
    original_forward = model.forward
    
    def tracked_forward(self, *args, **kwargs):
        result = original_forward(*args, **kwargs)
        
        # If model has setvae attribute, try to get latent variable information
        if hasattr(self, 'setvae') and hasattr(self.setvae, '_last_z_list'):
            self._last_z_list = self.setvae._last_z_list
            
        return result
        
    # Replace forward method
    model.forward = tracked_forward.__get__(model, model.__class__)
    
    return model


# Usage examples and test functions
def test_detector():
    """Test detector functionality"""
    detector = PosteriorCollapseDetector(
        kl_threshold=0.01,
        var_threshold=0.1,
        check_frequency=10,
        verbose=True
    )
    
    print("✅ Posterior collapse detector created successfully!")
    print(f"📁 Log directory: {detector.log_dir}")
    print(f"🔍 Detection parameters:")
    print(f"  - KL threshold: {detector.kl_threshold}")
    print(f"  - Variance threshold: {detector.var_threshold}")
    print(f"  - Check frequency: every {detector.check_frequency} steps")
    
    return detector

if __name__ == "__main__":
    test_detector()