#!/usr/bin/env python3
"""
Model Visualization Script for SeqSetVAE
Includes: latent space visualization, reconstruction quality, posterior collapse analysis
Modified to visualize only one sample (one patient) at a time
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
import config


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


def load_model(checkpoint_path, config):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
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
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def collect_single_sample_representations(model, dataloader, sample_idx=0):
    """
    Collect latent representations and reconstructions for a single sample (one patient)
    
    Args:
        model: The SeqSetVAE model
        dataloader: DataLoader with batch_size=1
        sample_idx: Index of the sample to visualize (default: 0)
    
    Returns:
        Dictionary containing the sample's representations
    """
    print(f"Collecting representations for sample {sample_idx}...")
    
    # Get the specific sample
    for i, batch in enumerate(dataloader):
        if i == sample_idx:
            break
    else:
        raise ValueError(f"Sample index {sample_idx} not found in dataloader")
    
    var = batch["var"]  # [1, num_events, embed_dim]
    val = batch["val"]  # [1, num_events, 1]
    time = batch["minute"]  # [1, num_events, 1]
    set_ids = batch["set_id"]  # [1, num_events, 1]
    labels = batch["label"]  # [1]
    
    print(f"Sample shape: var={var.shape}, val={val.shape}, time={time.shape}")
    print(f"Number of events: {var.shape[1]}")
    print(f"Label: {labels.item()}")
    
    with torch.no_grad():
        # Split into sets
        sets = model._split_sets(var, val, time, set_ids)
        
        # Process each sequence
        z_samples = []
        z_means = []
        z_logvars = []
        inputs = []
        times = []
        
        for s_dict in sets:
            var_s, val_s, time_s = s_dict["var"], s_dict["val"], s_dict["minute"]
            
            # Get latent representation
            _, z_list, _ = model.setvae(var_s, val_s)
            z_sample, mu, logvar = z_list[-1]  # Take the last level
            
            z_samples.append(z_sample.squeeze(1))
            z_means.append(mu.squeeze(1))
            z_logvars.append(logvar.squeeze(1))
            times.append(time_s.unique())
            
            # Compute input embedding
            if model.setvae.setvae.dim_reducer is not None:
                reduced = model.setvae.setvae.dim_reducer(var_s)
            else:
                reduced = var_s
            norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
            reduced_normalized = reduced / (norms + 1e-8)
            target_x = reduced_normalized * val_s
            inputs.append(target_x)
        
        # Stack to get sequences
        if z_samples:
            z_seq = torch.stack(z_samples, dim=1)  # [1, S, latent]
            
            # Create pos_tensor from times (similar to model's approach)
            pos_list = []
            for t in times:
                pos_list.append(t.unique().float())
            pos_tensor = torch.stack(pos_list, dim=1)  # [1, S]
            
            # Apply positional encoding and transformer
            z_seq_pos = model._apply_positional_encoding(z_seq, pos_tensor)
            S = z_seq.shape[1]
            h_seq = model.transformer(z_seq_pos, mask=model._create_causal_mask(S, z_seq.device))
            
            # Get reconstructions
            recons = []
            for idx, s_dict in enumerate(sets):
                N_t = s_dict["var"].size(1)
                recon = model.decoder(h_seq[:, idx], N_t)
                recons.append(recon)
    
    # Prepare results
    results = {
        'z_samples': z_seq.cpu().numpy() if z_samples else None,
        'z_means': torch.stack(z_means, dim=1).cpu().numpy() if z_means else None,
        'z_logvars': torch.stack(z_logvars, dim=1).cpu().numpy() if z_logvars else None,
        'h_seq': h_seq.cpu().numpy() if z_samples else None,
        'inputs': [x.cpu().numpy() for x in inputs],
        'recons': [x.cpu().numpy() for x in recons],
        'labels': labels.cpu().numpy(),
        'times': [t.cpu().numpy() for t in times],
        'set_ids': set_ids.cpu().numpy(),
    }
    
    return results


def analyze_single_sample_posterior_collapse(z_means, z_logvars, threshold=0.01):
    """Analyze posterior collapse for a single sample"""
    print("\nAnalyzing posterior collapse for single sample...")
    
    # Flatten sequences for analysis
    z_means_flat = z_means.reshape(-1, z_means.shape[-1])
    z_logvars_flat = z_logvars.reshape(-1, z_logvars.shape[-1])
    
    # Compute KL divergence for each dimension
    kl_per_dim = 0.5 * (z_means_flat**2 + np.exp(z_logvars_flat) - z_logvars_flat - 1)
    mean_kl_per_dim = kl_per_dim.mean(axis=0)
    
    # Identify collapsed dimensions
    collapsed_dims = mean_kl_per_dim < threshold
    n_collapsed = collapsed_dims.sum()
    
    # Compute variance per dimension
    var_per_dim = np.exp(z_logvars_flat).mean(axis=0)
    
    print(f"Total dimensions: {len(mean_kl_per_dim)}")
    print(f"Collapsed dimensions (KL < {threshold}): {n_collapsed} ({n_collapsed/len(mean_kl_per_dim)*100:.1f}%)")
    print(f"Mean KL divergence: {mean_kl_per_dim.mean():.6f}")
    print(f"Min/Max KL per dim: {mean_kl_per_dim.min():.6f} / {mean_kl_per_dim.max():.6f}")
    
    return {
        'kl_per_dim': mean_kl_per_dim,
        'var_per_dim': var_per_dim,
        'collapsed_dims': collapsed_dims,
        'n_collapsed': n_collapsed
    }


def plot_single_sample_visualizations(results, save_dir, sample_idx=0):
    """Create comprehensive visualizations for a single sample"""
    print(f"\nCreating visualizations for sample {sample_idx}...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Posterior Collapse Analysis
    if results['z_means'] is not None and results['z_logvars'] is not None:
        collapse_stats = analyze_single_sample_posterior_collapse(results['z_means'], results['z_logvars'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # KL divergence per dimension
        ax = axes[0, 0]
        ax.bar(range(len(collapse_stats['kl_per_dim'])), collapse_stats['kl_per_dim'])
        ax.axhline(y=0.01, color='r', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean KL Divergence')
        ax.set_title(f'KL Divergence per Latent Dimension (Sample {sample_idx})')
        ax.legend()
        
        # Variance per dimension
        ax = axes[0, 1]
        ax.bar(range(len(collapse_stats['var_per_dim'])), collapse_stats['var_per_dim'])
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean Variance')
        ax.set_title(f'Variance per Latent Dimension (Sample {sample_idx})')
        ax.set_yscale('log')
        
        # KL divergence distribution
        ax = axes[1, 0]
        ax.hist(collapse_stats['kl_per_dim'], bins=50, edgecolor='black')
        ax.axvline(x=0.01, color='r', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Number of Dimensions')
        ax.set_title(f'Distribution of KL Divergence (Sample {sample_idx})')
        ax.legend()
        
        # Active vs Collapsed dimensions pie chart
        ax = axes[1, 1]
        active_dims = len(collapse_stats['kl_per_dim']) - collapse_stats['n_collapsed']
        ax.pie([active_dims, collapse_stats['n_collapsed']], 
               labels=['Active', 'Collapsed'],
               autopct='%1.1f%%',
               colors=['green', 'red'])
        ax.set_title(f'Active vs Collapsed Latent Dimensions (Sample {sample_idx})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'posterior_collapse_analysis_sample_{sample_idx}.png'), dpi=300)
        plt.close()
    
    # 2. Latent Space Visualization (UMAP)
    if results['z_samples'] is not None and results['h_seq'] is not None:
        # Flatten for visualization
        z_flat = results['z_samples'].reshape(-1, results['z_samples'].shape[-1])
        h_flat = results['h_seq'].reshape(-1, results['h_seq'].shape[-1])
        
        # Fit UMAP
        print("Fitting UMAP for latent space...")
        all_latent = np.concatenate([z_flat, h_flat], axis=0)
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(all_latent)
        
        # Split embeddings
        n_z = len(z_flat)
        z_embed = embedding[:n_z]
        h_embed = embedding[n_z:]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Z space (before transformer)
        ax = axes[0]
        scatter = ax.scatter(z_embed[:, 0], z_embed[:, 1], 
                           c=range(len(z_embed)), alpha=0.6, s=50, cmap='viridis')
        ax.set_title(f'Latent Space Z - Sample {sample_idx} (before transformer)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax, label='Event Index')
        
        # H space (after transformer)
        ax = axes[1]
        scatter = ax.scatter(h_embed[:, 0], h_embed[:, 1],
                           c=range(len(h_embed)), alpha=0.6, s=50, cmap='viridis')
        ax.set_title(f'Latent Space H - Sample {sample_idx} (after transformer)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax, label='Event Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'latent_space_umap_sample_{sample_idx}.png'), dpi=300)
        plt.close()
    
    # 3. Reconstruction Quality
    if results['inputs'] and results['recons']:
        print("Analyzing reconstruction quality...")
        
        # Compute reconstruction errors
        recon_errors = []
        for inp, rec in zip(results['inputs'], results['recons']):
            if inp.shape == rec.shape:
                error = np.mean((inp - rec) ** 2)
                recon_errors.append(error)
        
        if recon_errors:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.hist(recon_errors, bins=20, edgecolor='black')
            ax.set_xlabel('Mean Squared Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of Reconstruction Errors - Sample {sample_idx}')
            ax.axvline(x=np.mean(recon_errors), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(recon_errors):.4f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'reconstruction_errors_sample_{sample_idx}.png'), dpi=300)
            plt.close()
    
    # 4. Latent Dimension Correlation Heatmap
    if results['z_means'] is not None:
        z_means_flat = results['z_means'].reshape(-1, results['z_means'].shape[-1])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(z_means_flat.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   vmin=-1, vmax=1, square=True)
        plt.title(f'Correlation Matrix of Latent Dimensions - Sample {sample_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'latent_correlation_sample_{sample_idx}.png'), dpi=300)
        plt.close()
    
    # 5. Time Series Visualization
    if results['times'] and results['z_means'] is not None:
        print("Creating time series visualization...")
        
        # Flatten times and z_means
        all_times = []
        all_z_means = []
        
        # Handle the case where times is a list of tensors and z_means is a 3D numpy array
        z_means_3d = results['z_means']  # Shape: [1, S, latent_dim]
        times_list = results['times']    # List of tensors
        
        # Flatten the 3D z_means array
        z_means_2d = z_means_3d.reshape(-1, z_means_3d.shape[-1])  # Shape: [S, latent_dim]
        
        # Process each time point and corresponding z_means
        for i, (times_tensor, z_means_row) in enumerate(zip(times_list, z_means_2d)):
            # Convert times_tensor to numpy if it's a tensor
            if hasattr(times_tensor, 'cpu'):
                times_flat = times_tensor.cpu().numpy().flatten()
            else:
                times_flat = np.array(times_tensor).flatten()
            
            # Repeat z_means_row for each time point in this set
            for time_val in times_flat:
                all_times.append(time_val)
                all_z_means.append(z_means_row)
        
        all_times = np.array(all_times)
        all_z_means = np.array(all_z_means)
        
        # Sort by time
        sort_idx = np.argsort(all_times)
        sorted_times = all_times[sort_idx]
        sorted_z_means = all_z_means[sort_idx]
        
        # Plot first few dimensions over time
        n_dims_to_plot = min(8, sorted_z_means.shape[1])
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(n_dims_to_plot):
            ax = axes[i]
            ax.plot(sorted_times, sorted_z_means[:, i], 'b-', linewidth=2)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f'Latent Dim {i}')
            ax.set_title(f'Latent Dimension {i} over Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'time_series_sample_{sample_idx}.png'), dpi=300)
        plt.close()
    
    print(f"\nVisualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize SeqSetVAE model for a single sample')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/patient_ehr",
                       help='Data directory path')
    parser.add_argument('--params_map_path', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/stats.csv",
                       help='Parameter mapping file path')
    parser.add_argument('--label_path', type=str,
                       default="/home/sunx/data/aiiih/data/mimic/processed/oc.csv",
                       help='Label file path')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations (default: auto-detect from checkpoint)')
    parser.add_argument('--config_name', type=str, default=None,
                       help='Config name for output directory (default: auto-detect from checkpoint)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of the sample to visualize (default: 0)')
    
    args = parser.parse_args()
    
    # Auto-detect config.name and save_dir if not provided
    if args.config_name is None or args.save_dir is None:
        # Try to extract config.name from checkpoint path
        checkpoint_parts = args.checkpoint.split(os.sep)
        config_name = None
        
        # Look for config.name in the checkpoint path
        for i, part in enumerate(checkpoint_parts):
            if part in ['SeqSetVAE-v2', 'SeqSetVAE-v1', 'SeqSetVAE']:  # Common config names
                config_name = part
                break
        
        if config_name is None:
            # Default config name
            config_name = "SeqSetVAE-v2"
        
        if args.config_name is None:
            args.config_name = config_name
        
        if args.save_dir is None:
            # Create save_dir based on config.name
            base_output_dir = os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else "."
            args.save_dir = os.path.join(base_output_dir, args.config_name, "visualizations")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"📁 Visualizations will be saved to: {args.save_dir}")
    print(f"🔍 Using config name: {args.config_name}")
    
    # Load configuration
    config = Config()
    
    # Load model
    model = load_model(args.checkpoint, config)
    
    # Prepare data with batch_size=1 for single sample visualization
    print("Loading data...")
    data_module = SeqSetVAEDataModule(
        args.data_dir, 
        args.params_map_path, 
        args.label_path,
        batch_size=1  # Force batch_size=1 for single sample visualization
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Collect representations for single sample
    results = collect_single_sample_representations(model, test_loader, args.sample_idx)
    
    # Create visualizations
    plot_single_sample_visualizations(results, args.save_dir, args.sample_idx)
    
    print(f"\nVisualization complete for sample {args.sample_idx}!")


if __name__ == "__main__":
    main()