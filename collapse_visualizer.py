import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import argparse
from collections import deque
import seaborn as sns

# Set matplotlib Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
            
        log_files = [f for f in os.listdir(self.log_dir) if f.startswith('collapse_detection_') and f.endswith('.log')]
        if not log_files:
            return None
            
        # Select the latest log file
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)), reverse=True)
        return os.path.join(self.log_dir, log_files[0])
        
    def parse_log_file(self, log_file: str):
        """Parse log file to extract data"""
        if not os.path.exists(log_file):
            return False
            
        # Check if file has been updated
        current_modified = os.path.getmtime(log_file)
        if current_modified <= self.last_modified:
            return False
            
        self.last_modified = current_modified
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Parse new log lines
            new_data_found = False
            
            for line in lines:
                if 'Step' in line and 'posterior collapse warning' in line:
                    # Parse warning information
                    try:
                        step_str = line.split('Step ')[1].split(':')[0]
                        step = int(step_str)
                        self.data['warnings'].append({
                            'step': step,
                            'message': line.strip(),
                            'time': datetime.now()
                        })
                        new_data_found = True
                    except:
                        pass
                        
                elif 'posterior collapse detected' in line:
                    # Parse collapse detection
                    try:
                        step_str = line.split('Step: ')[1]
                        step = int(step_str)
                        self.data['collapse_detected'] = True
                        self.data['collapse_step'] = step
                        new_data_found = True
                    except:
                        pass
                        
            return new_data_found
            
        except Exception as e:
            print(f"Failed to parse log file: {e}")
            return False
            
    def simulate_data(self):
        """Simulate data (for testing)"""
        if len(self.data['steps']) == 0:
            step = 0
        else:
            step = self.data['steps'][-1] + 1
            
        # Simulate data for different phases
        if step < 200:
            # Normal phase
            kl = 0.05 + np.random.normal(0, 0.01)
            var = 0.3 + np.random.normal(0, 0.05)
            active = 0.8 + np.random.normal(0, 0.1)
            recon = 2.0 + np.random.normal(0, 0.2)
        elif step < 400:
            # Beginning collapse
            progress = (step - 200) / 200
            kl = 0.05 * (1 - progress) + np.random.normal(0, 0.005)
            var = 0.3 * (1 - progress) + np.random.normal(0, 0.02)
            active = 0.8 * (1 - progress * 0.8) + np.random.normal(0, 0.05)
            recon = 2.0 + progress * 0.5 + np.random.normal(0, 0.1)
        else:
            # Complete collapse
            kl = 0.001 + np.random.normal(0, 0.0005)
            var = 0.01 + np.random.normal(0, 0.005)
            active = 0.05 + np.random.normal(0, 0.02)
            recon = 2.8 + np.random.normal(0, 0.1)
            
            if not self.data['collapse_detected'] and step > 420:
                self.data['collapse_detected'] = True
                self.data['collapse_step'] = step
                
        # Ensure values are within reasonable ranges
        kl = max(0, kl)
        var = max(0, var)
        active = np.clip(active, 0, 1)
        recon = max(0, recon)
        
        self.data['steps'].append(step)
        self.data['kl_divergence'].append(kl)
        self.data['variance'].append(var)
        self.data['active_ratio'].append(active)
        self.data['recon_loss'].append(recon)
        
        # Simulate warnings
        if step > 300 and len(self.data['warnings']) < 5 and np.random.random() < 0.1:
            self.data['warnings'].append({
                'step': step,
                'message': f'Step {step}: Low KL divergence warning',
                'time': datetime.now()
            })
            
    def update_plot(self, frame):
        """Update plots"""
        # Find and parse log file
        if self.log_file_path is None:
            self.log_file_path = self.find_log_file()
            
        if self.log_file_path:
            self.parse_log_file(self.log_file_path)
        else:
            # If no log file, use simulated data
            self.simulate_data()
            
        # If no data, return
        if len(self.data['steps']) == 0:
            return self.line_kl, self.line_var, self.line_active, self.line_recon
            
        steps = list(self.data['steps'])
        
        # Update KL divergence plot
        if self.data['kl_divergence']:
            kl_data = list(self.data['kl_divergence'])
            self.line_kl.set_data(steps, kl_data)
            self.ax_kl.relim()
            self.ax_kl.autoscale_view()
            
            # Add threshold line
            if 'kl' not in self.threshold_lines:
                self.threshold_lines['kl'] = self.ax_kl.axhline(
                    y=self.thresholds['kl_threshold'], 
                    color='red', linestyle='--', alpha=0.7, label='Threshold'
                )
                self.ax_kl.legend()
                
        # Update variance plot
        if self.data['variance']:
            var_data = list(self.data['variance'])
            self.line_var.set_data(steps, var_data)
            self.ax_var.relim()
            self.ax_var.autoscale_view()
            
            if 'var' not in self.threshold_lines:
                self.threshold_lines['var'] = self.ax_var.axhline(
                    y=self.thresholds['var_threshold'],
                    color='red', linestyle='--', alpha=0.7, label='Threshold'
                )
                self.ax_var.legend()
                
        # Update active units ratio plot
        if self.data['active_ratio']:
            active_data = list(self.data['active_ratio'])
            self.line_active.set_data(steps, active_data)
            self.ax_active.relim()
            self.ax_active.autoscale_view()
            
            if 'active' not in self.threshold_lines:
                self.threshold_lines['active'] = self.ax_active.axhline(
                    y=self.thresholds['active_threshold'],
                    color='red', linestyle='--', alpha=0.7, label='Threshold'
                )
                self.ax_active.legend()
                
        # Update reconstruction loss plot
        if self.data['recon_loss']:
            recon_data = list(self.data['recon_loss'])
            self.line_recon.set_data(steps, recon_data)
            self.ax_recon.relim()
            self.ax_recon.autoscale_view()
            
        # Update status panel
        self.update_status_panel()
        
        return self.line_kl, self.line_var, self.line_active, self.line_recon
        
    def update_status_panel(self):
        """Update status panel"""
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        # Current status
        current_time = datetime.now().strftime("%H:%M:%S")
        status_color = 'red' if self.data['collapse_detected'] else 'green'
        status_text = 'Posterior Collapse' if self.data['collapse_detected'] else 'Normal'
        
        # Status information
        status_info = [
            f"Current Time: {current_time}",
            f"Monitoring Status: {status_text}",
            f"Data Points: {len(self.data['steps'])}",
            f"Warning Count: {len(self.data['warnings'])}",
        ]
        
        if self.data['collapse_detected']:
            status_info.append(f"Collapse Detected at Step: {self.data['collapse_step']}")
            
        # Display status information
        for i, info in enumerate(status_info):
            self.ax_status.text(0.02, 0.8 - i*0.15, info, transform=self.ax_status.transAxes,
                              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor='lightblue', alpha=0.7))
        
        # Display recent warnings
        if self.data['warnings']:
            recent_warnings = self.data['warnings'][-3:]  # Show last 3 warnings
            warning_text = "Recent Warnings:\n" + "\n".join([
                f"Step {w['step']}: {w['message'].split(':')[-1].strip()}" 
                for w in recent_warnings
            ])
            
            self.ax_status.text(0.5, 0.8, warning_text, transform=self.ax_status.transAxes,
                              fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                              facecolor='yellow', alpha=0.7),
                              verticalalignment='top')
        
        # Add status indicator
        indicator_color = 'red' if self.data['collapse_detected'] else 'green'
        circle = plt.Circle((0.95, 0.5), 0.03, color=indicator_color, 
                          transform=self.ax_status.transAxes)
        self.ax_status.add_patch(circle)
        
    def start_monitoring(self):
        """Start monitoring"""
        print(f"🔍 Starting posterior collapse monitoring...")
        print(f"📁 Monitoring directory: {self.log_dir}")
        print(f"🔄 Update interval: {self.update_interval}ms")
        print("💡 Close window to stop monitoring")
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.update_interval,
            blit=False, cache_frame_data=False
        )
        
        # Display figure
        plt.tight_layout()
        plt.show()
        
        return ani

def create_dashboard(log_dir: str, update_interval: int = 1000):
    """Create monitoring dashboard"""
    visualizer = RealTimeCollapseVisualizer(log_dir, update_interval)
    return visualizer.start_monitoring()

def main():
    parser = argparse.ArgumentParser(description='VAE Posterior Collapse Real-time Visualization Monitor')
    
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Collapse detection log directory')
    parser.add_argument('--update_interval', type=int, default=2000,
                       help='Update interval (milliseconds)')
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode (use simulated data)')
    
    args = parser.parse_args()
    
    if args.demo:
        print("🎭 Demo mode: Using simulated data")
        args.log_dir = "./demo_logs"
        
    print("🚀 Starting VAE Posterior Collapse Real-time Monitor")
    print("=" * 50)
    
    try:
        ani = create_dashboard(args.log_dir, args.update_interval)
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred during monitoring: {e}")
        raise

if __name__ == "__main__":
    main()