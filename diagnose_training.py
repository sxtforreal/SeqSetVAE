#!/usr/bin/env python3
"""
Training Diagnosis Script for SeqSetVAE
Helps identify and fix training issues
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def analyze_gradients(model: nn.Module) -> Dict[str, Dict]:
    """
    Analyze gradient statistics for all model parameters
    """
    print("🔍 Analyzing gradient statistics...")
    
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            grad_min = grad.min().item()
            grad_max = grad.max().item()
            
            gradient_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'min': grad_min,
                'max': grad_max,
                'param_norm': param.data.norm().item(),
                'grad_param_ratio': grad_norm / (param.data.norm().item() + 1e-8)
            }
    
    return gradient_stats

def check_gradient_issues(gradient_stats: Dict[str, Dict]) -> List[str]:
    """
    Check for common gradient issues
    """
    issues = []
    
    # Check for gradient explosion
    for name, stats in gradient_stats.items():
        if stats['norm'] > 10.0:
            issues.append(f"🚨 Gradient explosion in {name}: norm={stats['norm']:.4f}")
        
        if stats['grad_param_ratio'] > 1.0:
            issues.append(f"⚠️  Large gradient/parameter ratio in {name}: ratio={stats['grad_param_ratio']:.4f}")
    
    # Check for gradient vanishing
    total_grad_norm = sum(stats['norm'] for stats in gradient_stats.values())
    if total_grad_norm < 1e-6:
        issues.append("🚨 Gradient vanishing detected: very small total gradient norm")
    
    return issues

def analyze_loss_components(model_outputs: Dict) -> Dict[str, float]:
    """
    Analyze individual loss components
    """
    print("📊 Analyzing loss components...")
    
    loss_components = {}
    
    # Extract loss components from model outputs
    if 'recon_loss' in model_outputs:
        loss_components['reconstruction'] = model_outputs['recon_loss'].item()
    
    if 'kl_loss' in model_outputs:
        loss_components['kl_divergence'] = model_outputs['kl_loss'].item()
    
    if 'classification_loss' in model_outputs:
        loss_components['classification'] = model_outputs['classification_loss'].item()
    
    if 'total_loss' in model_outputs:
        loss_components['total'] = model_outputs['total_loss'].item()
    
    return loss_components

def check_loss_balance(loss_components: Dict[str, float]) -> List[str]:
    """
    Check if loss components are balanced
    """
    issues = []
    
    if 'kl_divergence' in loss_components and 'reconstruction' in loss_components:
        kl_recon_ratio = loss_components['kl_divergence'] / (loss_components['reconstruction'] + 1e-8)
        
        if kl_recon_ratio > 10.0:
            issues.append(f"🚨 KL divergence dominates reconstruction loss: ratio={kl_recon_ratio:.4f}")
        elif kl_recon_ratio < 0.01:
            issues.append(f"⚠️  KL divergence too small compared to reconstruction: ratio={kl_recon_ratio:.4f}")
    
    if 'classification' in loss_components and 'total_loss' in loss_components:
        cls_ratio = loss_components['classification'] / loss_components['total_loss']
        if cls_ratio > 0.9:
            issues.append(f"⚠️  Classification loss dominates total loss: ratio={cls_ratio:.4f}")
    
    return issues

def analyze_parameter_distributions(model: nn.Module) -> Dict[str, Dict]:
    """
    Analyze parameter value distributions
    """
    print("📈 Analyzing parameter distributions...")
    
    param_stats = {}
    
    for name, param in model.named_parameters():
        data = param.data.cpu().numpy().flatten()
        
        param_stats[name] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'zeros': np.sum(data == 0),
            'total': len(data),
            'zero_ratio': np.sum(data == 0) / len(data)
        }
    
    return param_stats

def check_parameter_issues(param_stats: Dict[str, Dict]) -> List[str]:
    """
    Check for parameter-related issues
    """
    issues = []
    
    for name, stats in param_stats.items():
        # Check for parameter explosion
        if abs(stats['mean']) > 100.0:
            issues.append(f"🚨 Parameter explosion in {name}: mean={stats['mean']:.4f}")
        
        # Check for parameter vanishing
        if stats['std'] < 1e-6:
            issues.append(f"⚠️  Parameter vanishing in {name}: std={stats['std']:.2e}")
        
        # Check for too many zero parameters
        if stats['zero_ratio'] > 0.8:
            issues.append(f"⚠️  Too many zero parameters in {name}: {stats['zero_ratio']:.1%}")
    
    return issues

def generate_diagnostic_report(model: nn.Module, 
                             gradient_stats: Dict[str, Dict],
                             loss_components: Dict[str, float],
                             param_stats: Dict[str, Dict]) -> str:
    """
    Generate a comprehensive diagnostic report
    """
    report = []
    report.append("=" * 80)
    report.append("🔍 SEQSETVAE TRAINING DIAGNOSTIC REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    report.append("📊 MODEL SUMMARY:")
    report.append(f"   - Total parameters: {total_params:,}")
    report.append(f"   - Trainable parameters: {trainable_params:,}")
    report.append(f"   - Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    report.append("")
    
    # Gradient analysis
    report.append("📈 GRADIENT ANALYSIS:")
    if gradient_stats:
        total_grad_norm = sum(stats['norm'] for stats in gradient_stats.values())
        avg_grad_norm = total_grad_norm / len(gradient_stats)
        
        report.append(f"   - Total gradient norm: {total_grad_norm:.4f}")
        report.append(f"   - Average gradient norm: {avg_grad_norm:.4f}")
        report.append(f"   - Parameters with gradients: {len(gradient_stats)}")
        
        # Top 5 largest gradients
        sorted_grads = sorted(gradient_stats.items(), key=lambda x: x[1]['norm'], reverse=True)
        report.append("   - Top 5 largest gradients:")
        for i, (name, stats) in enumerate(sorted_grads[:5]):
            report.append(f"     {i+1}. {name}: {stats['norm']:.4f}")
    else:
        report.append("   - No gradients available (model not trained yet)")
    report.append("")
    
    # Loss analysis
    report.append("💔 LOSS ANALYSIS:")
    if loss_components:
        for name, value in loss_components.items():
            report.append(f"   - {name}: {value:.6f}")
    else:
        report.append("   - No loss components available")
    report.append("")
    
    # Parameter analysis
    report.append("🔧 PARAMETER ANALYSIS:")
    if param_stats:
        total_zeros = sum(stats['zeros'] for stats in param_stats.values())
        total_params_count = sum(stats['total'] for stats in param_stats.values())
        overall_zero_ratio = total_zeros / total_params_count
        
        report.append(f"   - Overall zero parameter ratio: {overall_zero_ratio:.2%}")
        report.append(f"   - Total zero parameters: {total_zeros:,}")
        
        # Top 5 largest parameter norms
        sorted_params = sorted(param_stats.items(), key=lambda x: x[1]['std'], reverse=True)
        report.append("   - Top 5 largest parameter stds:")
        for i, (name, stats) in enumerate(sorted_params[:5]):
            report.append(f"     {i+1}. {name}: std={stats['std']:.4f}")
    else:
        report.append("   - No parameter statistics available")
    report.append("")
    
    # Issues summary
    gradient_issues = check_gradient_issues(gradient_stats)
    loss_issues = check_loss_balance(loss_components)
    param_issues = check_parameter_issues(param_stats)
    
    all_issues = gradient_issues + loss_issues + param_issues
    
    report.append("🚨 ISSUES DETECTED:")
    if all_issues:
        for issue in all_issues:
            report.append(f"   {issue}")
    else:
        report.append("   ✅ No major issues detected")
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS:")
    if gradient_issues:
        report.append("   - Consider reducing learning rate")
        report.append("   - Implement gradient clipping")
        report.append("   - Check for exploding gradients in specific layers")
    
    if loss_issues:
        report.append("   - Adjust loss component weights")
        report.append("   - Check KL divergence annealing")
        report.append("   - Balance reconstruction vs KL loss")
    
    if param_issues:
        report.append("   - Check weight initialization")
        report.append("   - Reduce model complexity if needed")
        report.append("   - Consider regularization techniques")
    
    if not all_issues:
        report.append("   - Training appears stable")
        report.append("   - Monitor for gradual improvements")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def plot_training_diagnostics(gradient_stats: Dict[str, Dict],
                            loss_components: Dict[str, float],
                            param_stats: Dict[str, Dict]):
    """
    Create diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SeqSetVAE Training Diagnostics', fontsize=16)
    
    # Gradient norm distribution
    if gradient_stats:
        grad_norms = [stats['norm'] for stats in gradient_stats.values()]
        axes[0, 0].hist(grad_norms, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Gradient Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Gradient Norm Distribution')
        axes[0, 0].set_yscale('log')
    
    # Loss components
    if loss_components:
        names = list(loss_components.keys())
        values = list(loss_components.values())
        axes[0, 1].bar(names, values, color='lightcoral')
        axes[0, 1].set_ylabel('Loss Value')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Parameter std distribution
    if param_stats:
        param_stds = [stats['std'] for stats in param_stats.values()]
        axes[1, 0].hist(param_stds, bins=30, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('Parameter Standard Deviation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Parameter Std Distribution')
        axes[1, 0].set_yscale('log')
    
    # Zero parameter ratio
    if param_stats:
        zero_ratios = [stats['zero_ratio'] for stats in param_stats.values()]
        axes[1, 1].hist(zero_ratios, bins=30, alpha=0.7, color='gold')
        axes[1, 1].set_xlabel('Zero Parameter Ratio')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Zero Parameter Ratio Distribution')
    
    plt.tight_layout()
    plt.savefig('training_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Diagnostic plots saved as 'training_diagnostics.png'")

def main():
    """
    Main diagnostic function
    """
    print("🔍 SeqSetVAE Training Diagnostics")
    print("=" * 50)
    
    # This script is designed to be used with a trained model
    # You would typically call it after training or during training
    
    print("📝 To use this diagnostic script:")
    print("   1. Import it in your training script")
    print("   2. Call the diagnostic functions after training steps")
    print("   3. Or use it interactively with a trained model")
    print("")
    
    print("💡 Example usage:")
    print("   from diagnose_training import *")
    print("   ")
    print("   # After training step")
    print("   gradient_stats = analyze_gradients(model)")
    print("   issues = check_gradient_issues(gradient_stats)")
    print("   ")
    print("   # Generate report")
    print("   report = generate_diagnostic_report(...)")
    print("   print(report)")
    
    print("")
    print("🚀 For immediate help with high training loss:")
    print("   1. Use config_optimized.py for stable training")
    print("   2. Run train_stable.py instead of train.py")
    print("   3. Monitor the diagnostic output during training")

if __name__ == "__main__":
    main()