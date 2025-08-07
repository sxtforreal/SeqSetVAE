#!/usr/bin/env python3
"""
Test script for the simplified PosteriorMetricsMonitor
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from posterior_collapse_detector import PosteriorMetricsMonitor

def test_metrics_monitor():
    """Test the PosteriorMetricsMonitor functionality"""
    
    print("🧪 Testing PosteriorMetricsMonitor...")
    
    # Create monitor
    monitor = PosteriorMetricsMonitor(
        update_frequency=10,
        plot_frequency=50,
        window_size=20,
        log_dir="./test_metrics",
        verbose=True
    )
    
    print(f"✅ Monitor created successfully:")
    print(f"  - Update frequency: {monitor.update_frequency}")
    print(f"  - Plot frequency: {monitor.plot_frequency}")
    print(f"  - Window size: {monitor.window_size}")
    print(f"  - Log directory: {monitor.log_dir}")
    
    # Simulate some metrics
    print("\n📊 Simulating metrics updates...")
    
    for step in range(1, 101):
        monitor.global_step = step
        
        # Simulate metrics
        metrics = {
            'kl_divergence': 0.1 * np.exp(-step / 50) + 0.001,  # Decreasing KL
            'variance': 0.5 * np.exp(-step / 30) + 0.01,        # Decreasing variance
            'active_units': 0.8 * np.exp(-step / 40) + 0.1,     # Decreasing active units
            'reconstruction_loss': 2.0 + 0.5 * np.sin(step / 10), # Oscillating recon loss
        }
        
        # Update history
        monitor.update_history(metrics)
        
        # Print progress
        if step % 20 == 0:
            print(f"  Step {step}: KL={metrics['kl_divergence']:.6f}, "
                  f"Var={metrics['variance']:.6f}, Active={metrics['active_units']:.3f}")
    
    # Save a test plot
    print("\n📈 Saving test plot...")
    monitor.save_metrics_plot()
    
    print("\n✅ Test completed successfully!")
    print(f"📁 Check the test plot in: {monitor.log_dir}")

if __name__ == "__main__":
    test_metrics_monitor()