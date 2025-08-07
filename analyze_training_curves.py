#!/usr/bin/env python3
"""
Training Curves Analysis Script
Extracts and visualizes metrics from TensorBoard logs
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
warnings.filterwarnings('ignore')


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
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 3. Reconstruction Loss
    ax = axes[1, 0]
    if 'train_recon_step' in metrics_df:
        ax.plot(metrics_df['train_recon_step']['steps'], 
                metrics_df['train_recon_step']['values'], 
                label='Train Recon', alpha=0.8)
    elif 'train_recon' in metrics_df:
        ax.plot(metrics_df['train_recon']['steps'], 
                metrics_df['train_recon']['values'], 
                label='Train Recon', alpha=0.8)
    if 'val_recon' in metrics_df:
        ax.plot(metrics_df['val_recon']['steps'], 
                metrics_df['val_recon']['values'], 
                label='Val Recon', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Prediction Loss
    ax = axes[1, 1]
    if 'train_pred_step' in metrics_df:
        ax.plot(metrics_df['train_pred_step']['steps'], 
                metrics_df['train_pred_step']['values'], 
                label='Train Pred', alpha=0.8)
    elif 'train_pred' in metrics_df:
        ax.plot(metrics_df['train_pred']['steps'], 
                metrics_df['train_pred']['values'], 
                label='Train Pred', alpha=0.8)
    if 'val_pred' in metrics_df:
        ax.plot(metrics_df['val_pred']['steps'], 
                metrics_df['val_pred']['values'], 
                label='Val Pred', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Prediction Loss')
    ax.set_title('Prediction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
    plt.close()


def plot_performance_metrics(metrics_df, save_dir):
    """Plot performance metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. AUC
    ax = axes[0]
    if 'val_auc' in metrics_df:
        ax.plot(metrics_df['val_auc']['steps'], 
                metrics_df['val_auc']['values'], 
                label='Val AUC', color='green', linewidth=2)
        # Mark best AUC
        best_idx = np.argmax(metrics_df['val_auc']['values'])
        best_step = metrics_df['val_auc']['steps'].iloc[best_idx]
        best_auc = metrics_df['val_auc']['values'].iloc[best_idx]
        ax.scatter(best_step, best_auc, color='red', s=100, zorder=5)
        ax.text(best_step, best_auc, f'Best: {best_auc:.4f}', 
                ha='right', va='bottom')
    ax.set_xlabel('Steps')
    ax.set_ylabel('AUC')
    ax.set_title('Validation AUC')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # 2. AUPRC
    ax = axes[1]
    if 'val_auprc' in metrics_df:
        ax.plot(metrics_df['val_auprc']['steps'], 
                metrics_df['val_auprc']['values'], 
                label='Val AUPRC', color='blue', linewidth=2)
        # Mark best AUPRC
        best_idx = np.argmax(metrics_df['val_auprc']['values'])
        best_step = metrics_df['val_auprc']['steps'].iloc[best_idx]
        best_auprc = metrics_df['val_auprc']['values'].iloc[best_idx]
        ax.scatter(best_step, best_auprc, color='red', s=100, zorder=5)
        ax.text(best_step, best_auprc, f'Best: {best_auprc:.4f}', 
                ha='right', va='bottom')
    ax.set_xlabel('Steps')
    ax.set_ylabel('AUPRC')
    ax.set_title('Validation AUPRC')
    ax.grid(True, alpha=0.3)
    
    # 3. Accuracy
    ax = axes[2]
    if 'val_accuracy' in metrics_df:
        ax.plot(metrics_df['val_accuracy']['steps'], 
                metrics_df['val_accuracy']['values'], 
                label='Val Accuracy', color='orange', linewidth=2)
        # Mark best accuracy
        best_idx = np.argmax(metrics_df['val_accuracy']['values'])
        best_step = metrics_df['val_accuracy']['steps'].iloc[best_idx]
        best_acc = metrics_df['val_accuracy']['values'].iloc[best_idx]
        ax.scatter(best_step, best_acc, color='red', s=100, zorder=5)
        ax.text(best_step, best_acc, f'Best: {best_acc:.4f}', 
                ha='right', va='bottom')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), dpi=300)
    plt.close()


def plot_training_dynamics(metrics_df, save_dir):
    """Plot training dynamics (beta, weights)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Beta annealing
    ax = axes[0, 0]
    if 'train_beta_step' in metrics_df:
        ax.plot(metrics_df['train_beta_step']['steps'], 
                metrics_df['train_beta_step']['values'], 
                label='Beta', color='purple', linewidth=2)
    elif 'train_beta' in metrics_df:
        ax.plot(metrics_df['train_beta']['steps'], 
                metrics_df['train_beta']['values'], 
                label='Beta', color='purple', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Beta Value')
    ax.set_title('Beta Annealing Schedule')
    ax.grid(True, alpha=0.3)
    
    # 2. Loss weights
    ax = axes[0, 1]
    if 'train_recon_weight_step' in metrics_df:
        ax.plot(metrics_df['train_recon_weight_step']['steps'], 
                metrics_df['train_recon_weight_step']['values'], 
                label='Recon Weight', alpha=0.8)
    elif 'train_recon_weight' in metrics_df:
        ax.plot(metrics_df['train_recon_weight']['steps'], 
                metrics_df['train_recon_weight']['values'], 
                label='Recon Weight', alpha=0.8)
    if 'train_pred_weight_step' in metrics_df:
        ax.plot(metrics_df['train_pred_weight_step']['steps'], 
                metrics_df['train_pred_weight_step']['values'], 
                label='Pred Weight', alpha=0.8)
    elif 'train_pred_weight' in metrics_df:
        ax.plot(metrics_df['train_pred_weight']['steps'], 
                metrics_df['train_pred_weight']['values'], 
                label='Pred Weight', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Weight')
    ax.set_title('Loss Component Weights')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. KL vs Reconstruction trade-off
    ax = axes[1, 0]
    if 'train_kl_step' in metrics_df and 'train_recon_step' in metrics_df:
        # Sample points to avoid overcrowding
        sample_idx = np.linspace(0, len(metrics_df['train_kl_step'])-1, 100, dtype=int)
        kl_values = metrics_df['train_kl_step']['values'].iloc[sample_idx]
        recon_values = metrics_df['train_recon_step']['values'].iloc[sample_idx]
        
        scatter = ax.scatter(recon_values, kl_values, 
                           c=sample_idx, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Training Progress')
    elif 'train_kl' in metrics_df and 'train_recon' in metrics_df:
        # Sample points to avoid overcrowding
        sample_idx = np.linspace(0, len(metrics_df['train_kl'])-1, 100, dtype=int)
        kl_values = metrics_df['train_kl']['values'].iloc[sample_idx]
        recon_values = metrics_df['train_recon']['values'].iloc[sample_idx]
        
        scatter = ax.scatter(recon_values, kl_values, 
                           c=sample_idx, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Training Progress')
        
    ax.set_xlabel('Reconstruction Loss')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL-Reconstruction Trade-off')
    ax.grid(True, alpha=0.3)
    
    # 4. Loss composition over time
    ax = axes[1, 1]
    if all(k in metrics_df for k in ['train_kl_step', 'train_recon_step', 'train_pred_step']):
        steps = metrics_df['train_kl_step']['steps']
        kl = metrics_df['train_kl_step']['values']
        recon = metrics_df['train_recon_step']['values']
        pred = metrics_df['train_pred_step']['values']
        
        # Normalize by total
        total = kl + recon + pred + 1e-8
        kl_norm = kl / total
        recon_norm = recon / total
        pred_norm = pred / total
        
        ax.fill_between(steps, 0, kl_norm, alpha=0.7, label='KL')
        ax.fill_between(steps, kl_norm, kl_norm + recon_norm, 
                       alpha=0.7, label='Recon')
        ax.fill_between(steps, kl_norm + recon_norm, 1, 
                       alpha=0.7, label='Pred')
    elif all(k in metrics_df for k in ['train_kl', 'train_recon', 'train_pred']):
        steps = metrics_df['train_kl']['steps']
        kl = metrics_df['train_kl']['values']
        recon = metrics_df['train_recon']['values']
        pred = metrics_df['train_pred']['values']
        
        # Normalize by total
        total = kl + recon + pred + 1e-8
        kl_norm = kl / total
        recon_norm = recon / total
        pred_norm = pred / total
        
        ax.fill_between(steps, 0, kl_norm, alpha=0.7, label='KL')
        ax.fill_between(steps, kl_norm, kl_norm + recon_norm, 
                       alpha=0.7, label='Recon')
        ax.fill_between(steps, kl_norm + recon_norm, 1, 
                       alpha=0.7, label='Pred')
        
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss Proportion')
    ax.set_title('Loss Composition Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_dynamics.png'), dpi=300)
    plt.close()


def analyze_collapse_indicators(metrics_df, save_dir):
    """Analyze potential collapse indicators"""
    print("\n=== Collapse Analysis ===")
    
    # Check final KL divergence
    kl_metric = 'train_kl_step' if 'train_kl_step' in metrics_df else 'train_kl'
    if kl_metric in metrics_df:
        final_kl = metrics_df[kl_metric]['values'].iloc[-1]
        avg_kl_last_100 = metrics_df[kl_metric]['values'].iloc[-100:].mean()
        print(f"Final KL divergence: {final_kl:.6f}")
        print(f"Average KL (last 100 steps): {avg_kl_last_100:.6f}")
        
        if avg_kl_last_100 < 0.01:
            print("⚠️  WARNING: Very low KL divergence detected - possible posterior collapse!")
        elif avg_kl_last_100 < 0.1:
            print("⚠️  CAUTION: Low KL divergence - monitor for collapse")
        else:
            print("✅ KL divergence appears healthy")
    
    # Check performance plateau
    if 'val_auc' in metrics_df:
        auc_values = metrics_df['val_auc']['values']
        if len(auc_values) > 20:
            recent_auc = auc_values.iloc[-10:].mean()
            earlier_auc = auc_values.iloc[-20:-10].mean()
            improvement = recent_auc - earlier_auc
            
            print(f"\nValidation AUC - Recent: {recent_auc:.4f}, Earlier: {earlier_auc:.4f}")
            print(f"Improvement: {improvement:.4f}")
            
            if abs(improvement) < 0.001:
                print("⚠️  Performance has plateaued")
    
    # Save analysis summary
    with open(os.path.join(save_dir, 'collapse_analysis.txt'), 'w') as f:
        f.write("=== Collapse Analysis Summary ===\n\n")
        
        if kl_metric in metrics_df:
            f.write(f"Final KL divergence: {final_kl:.6f}\n")
            f.write(f"Average KL (last 100 steps): {avg_kl_last_100:.6f}\n")
            
            # KL trend
            kl_values = metrics_df[kl_metric]['values']
            kl_start = kl_values.iloc[:100].mean()
            kl_end = kl_values.iloc[-100:].mean()
            f.write(f"KL trend: {kl_start:.6f} -> {kl_end:.6f} ")
            f.write(f"(reduction: {(1 - kl_end/kl_start)*100:.1f}%)\n")
        
        if 'val_auc' in metrics_df:
            best_auc = metrics_df['val_auc']['values'].max()
            final_auc = metrics_df['val_auc']['values'].iloc[-1]
            f.write(f"\nBest validation AUC: {best_auc:.4f}\n")
            f.write(f"Final validation AUC: {final_auc:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze training curves from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Path to TensorBoard log directory')
    parser.add_argument('--save_dir', type=str, default='./training_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load metrics
    try:
        metrics_df = load_tensorboard_logs(args.log_dir)
        print(f"\nLoaded {len(metrics_df)} metrics")
        print("Available metrics:", list(metrics_df.keys()))
    except Exception as e:
        print(f"Error loading logs: {e}")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_loss_curves(metrics_df, args.save_dir)
    plot_performance_metrics(metrics_df, args.save_dir)
    plot_training_dynamics(metrics_df, args.save_dir)
    
    # Analyze collapse indicators
    analyze_collapse_indicators(metrics_df, args.save_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.save_dir}")
    print("\nKey files generated:")
    print("- loss_curves.png: Training and validation losses")
    print("- performance_metrics.png: AUC, AUPRC, Accuracy")
    print("- training_dynamics.png: Beta schedule, loss weights, trade-offs")
    print("- collapse_analysis.txt: Summary of collapse indicators")


if __name__ == "__main__":
    main()