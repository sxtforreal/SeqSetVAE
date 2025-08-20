#!/usr/bin/env python3
"""
Demo script for the recon eval JSON visualizer
Creates sample data and demonstrates visualization capabilities
"""

import json
import os
import numpy as np
from datetime import datetime
from recon_eval_visualizer import ReconEvalVisualizer, compare_multiple_files

def create_sample_json(filename: str, quality_level: str = "good"):
    """Create a sample JSON file for demonstration"""
    
    # Base metrics that vary by quality level
    if quality_level == "excellent":
        base_metrics = {
            "nn_l2_mean": 0.15,
            "nn_l2_median": 0.12,
            "nn_l2_p95": 0.35,
            "dir_cosine_mean": 0.95,
            "mag_mae": 0.08,
            "mag_rmse": 0.12,
            "mag_corr": 0.92,
            "scale_ratio": 1.02,
            "chamfer_l2_mean": 0.08,
            "coverage@0.25": 0.85,
            "coverage@0.5": 0.95,
            "coverage@1.0": 0.98,
            "coverage@2.0": 1.0,
            "orig_norm_mean": 1.2,
            "recon_norm_mean": 1.22,
            "orig_norm_std": 0.3,
            "recon_norm_std": 0.31
        }
    elif quality_level == "good":
        base_metrics = {
            "nn_l2_mean": 0.45,
            "nn_l2_median": 0.38,
            "nn_l2_p95": 0.85,
            "dir_cosine_mean": 0.78,
            "mag_mae": 0.25,
            "mag_rmse": 0.35,
            "mag_corr": 0.75,
            "scale_ratio": 1.15,
            "chamfer_l2_mean": 0.28,
            "coverage@0.25": 0.65,
            "coverage@0.5": 0.78,
            "coverage@1.0": 0.92,
            "coverage@2.0": 0.98,
            "orig_norm_mean": 1.2,
            "recon_norm_mean": 1.38,
            "orig_norm_std": 0.3,
            "recon_norm_std": 0.42
        }
    else:  # poor
        base_metrics = {
            "nn_l2_mean": 1.25,
            "nn_l2_median": 1.08,
            "nn_l2_p95": 2.15,
            "dir_cosine_mean": 0.45,
            "mag_mae": 0.85,
            "mag_rmse": 1.15,
            "mag_corr": 0.35,
            "scale_ratio": 1.65,
            "chamfer_l2_mean": 0.95,
            "coverage@0.25": 0.25,
            "coverage@0.5": 0.45,
            "coverage@1.0": 0.68,
            "coverage@2.0": 0.85,
            "orig_norm_mean": 1.2,
            "recon_norm_mean": 1.98,
            "orig_norm_std": 0.3,
            "recon_norm_std": 0.75
        }
    
    # Add num_samples
    base_metrics["num_samples"] = 5
    
    # Create per-sample details with some variation
    details = []
    for i in range(5):
        # Add some random variation to base metrics
        sample_metrics = {}
        for key, value in base_metrics.items():
            if key == "num_samples":
                continue
            if isinstance(value, (int, float)):
                # Add ±10% random variation
                variation = np.random.uniform(0.9, 1.1)
                sample_metrics[key] = value * variation
            else:
                sample_metrics[key] = value
        
        # Add per-set information (simplified)
        per_set_data = []
        for j in range(3):  # 3 sets per sample
            set_metrics = {k: v * np.random.uniform(0.8, 1.2) 
                          for k, v in sample_metrics.items() 
                          if isinstance(v, (int, float))}
            set_metrics.update({
                "set_index": j,
                "n_recon": np.random.randint(50, 200),
                "n_orig": np.random.randint(50, 200)
            })
            per_set_data.append(set_metrics)
        
        details.append({
            "sample_index": i,
            "global": sample_metrics,
            "per_set": per_set_data
        })
    
    # Create final JSON structure
    json_data = {
        "summary": base_metrics,
        "details": details
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Created sample JSON: {filename}")
    return filename

def demo_visualization():
    """Demonstrate the visualization capabilities"""
    print("🎨 Recon Eval JSON Visualizer Demo")
    print("=" * 50)
    
    # Create demo directory
    demo_dir = "./demo_recon_eval"
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create sample JSON files with different quality levels
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    excellent_file = create_sample_json(
        os.path.join(demo_dir, f"reconstruction_eval_excellent_{ts}.json"), 
        "excellent"
    )
    
    good_file = create_sample_json(
        os.path.join(demo_dir, f"reconstruction_eval_good_{ts}.json"), 
        "good"
    )
    
    poor_file = create_sample_json(
        os.path.join(demo_dir, f"reconstruction_eval_poor_{ts}.json"), 
        "poor"
    )
    
    print(f"\n🎯 Demonstrating single file visualization...")
    
    # Visualize the "good" quality file
    visualizer = ReconEvalVisualizer(good_file)
    visualizer.print_summary()
    visualizer.create_comparison_table()
    
    # Create visualizations
    vis_dir = "./demo_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    dashboard_path = os.path.join(vis_dir, f"demo_dashboard_{ts}.png")
    detailed_path = os.path.join(vis_dir, f"demo_detailed_{ts}.png")
    report_path = os.path.join(vis_dir, f"demo_report_{ts}.md")
    
    print(f"\n📊 Creating dashboard visualization...")
    visualizer.create_summary_dashboard(dashboard_path, show=False)
    
    print(f"📈 Creating detailed analysis...")
    visualizer.create_detailed_analysis(detailed_path, show=False)
    
    print(f"📄 Exporting summary report...")
    visualizer.export_summary_report(report_path)
    
    # Demonstrate comparison
    print(f"\n🔄 Demonstrating multi-file comparison...")
    comparison_path = os.path.join(vis_dir, f"demo_comparison_{ts}.png")
    compare_multiple_files([excellent_file, good_file, poor_file], comparison_path, show=False)
    
    print(f"\n✅ Demo completed! Check the following outputs:")
    print(f"  📊 Dashboard: {dashboard_path}")
    print(f"  📈 Detailed: {detailed_path}")
    print(f"  📄 Report: {report_path}")
    print(f"  🔄 Comparison: {comparison_path}")
    
    print(f"\n💡 Usage examples:")
    print(f"  # Visualize single file:")
    print(f"  python recon_eval_visualizer.py --json_file {good_file}")
    print(f"  ")
    print(f"  # Compare multiple files:")
    print(f"  python recon_eval_visualizer.py --json_dir {demo_dir} --compare")
    print(f"  ")
    print(f"  # Export report without showing plots:")
    print(f"  python recon_eval_visualizer.py --json_file {good_file} --export_report --no_show")

if __name__ == "__main__":
    demo_visualization()