#!/usr/bin/env python3
"""
Model Visualization Script for SeqSetVAE
Includes: latent space visualization, reconstruction quality, posterior collapse analysis
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
from train_config import Config


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


def collect_latent_representations(model, dataloader, max_batches=10):
    """Collect latent representations and reconstructions"""
    print("Collecting latent representations...")
    
    all_z_samples = []  # Latent codes from encoder
    all_z_means = []    # Latent means
    all_z_logvars = []  # Latent log variances
    all_h_seq = []      # Transformer outputs
    all_inputs = []     # Original inputs
    all_recons = []     # Reconstructions
    all_labels = []     # Labels
    all_times = []      # Time information
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch_idx >= max_batches:
                break
                
            var = batch["var"]
            val = batch["val"]
            time = batch["minute"]
            set_ids = batch["set_id"]
            labels = batch["label"]
            
            # Split into sets
            sets = model._split_sets(var, val, time, set_ids)
            
            # Process each sequence
            batch_z_samples = []
            batch_z_means = []
            batch_z_logvars = []
            batch_inputs = []
            batch_times = []
            
            for s_dict in sets:
                var_s, val_s, time_s = s_dict["var"], s_dict["val"], s_dict["minute"]
                
                # Get latent representation
                _, z_list, _ = model.setvae(var_s, val_s)
                z_sample, mu, logvar = z_list[-1]  # Take the last level
                
                batch_z_samples.append(z_sample.squeeze(1))
                batch_z_means.append(mu.squeeze(1))
                batch_z_logvars.append(logvar.squeeze(1))
                batch_times.append(time_s.unique())
                
                # Compute input embedding
                if model.setvae.setvae.dim_reducer is not None:
                    reduced = model.setvae.setvae.dim_reducer(var_s)
                else:
                    reduced = var_s
                norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
                reduced_normalized = reduced / (norms + 1e-8)
                target_x = reduced_normalized * val_s
                batch_inputs.append(target_x)
            
            # Stack to get sequences
            if batch_z_samples:
                z_seq = torch.stack(batch_z_samples, dim=1)  # [B, S, latent]
                pos_tensor = torch.stack(batch_times, dim=1)  # [B, S]
                
                # Apply RoPE and transformer
                z_seq_rope = model._apply_rope(z_seq, pos_tensor)
                S = z_seq.shape[1]
                h_seq = model.transformer(z_seq_rope, mask=model._causal_mask(S, z_seq.device))
                
                # Collect data
                all_z_samples.append(z_seq)
                all_z_means.append(torch.stack(batch_z_means, dim=1))
                all_z_logvars.append(torch.stack(batch_z_logvars, dim=1))
                all_h_seq.append(h_seq)
                all_labels.extend([labels] * z_seq.shape[1])
                
                # Get reconstructions
                batch_recons = []
                for idx, s_dict in enumerate(sets):
                    N_t = s_dict["var"].size(1)
                    recon = model.decoder(h_seq[:, idx], N_t)
                    batch_recons.append(recon)
                all_recons.extend(batch_recons)
                all_inputs.extend(batch_inputs)
    
    # Concatenate all data
    results = {
        'z_samples': torch.cat(all_z_samples, dim=0).cpu().numpy() if all_z_samples else None,
        'z_means': torch.cat(all_z_means, dim=0).cpu().numpy() if all_z_means else None,
        'z_logvars': torch.cat(all_z_logvars, dim=0).cpu().numpy() if all_z_logvars else None,
        'h_seq': torch.cat(all_h_seq, dim=0).cpu().numpy() if all_h_seq else None,
        'inputs': [x.cpu().numpy() for x in all_inputs],
        'recons': [x.cpu().numpy() for x in all_recons],
        'labels': torch.tensor(all_labels).cpu().numpy() if all_labels else None,
    }
    
    return results


def analyze_posterior_collapse(z_means, z_logvars, threshold=0.01):
    """Analyze posterior collapse in latent dimensions"""
    print("\nAnalyzing posterior collapse...")
    
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


def plot_latent_space(results, save_dir):
    """Create comprehensive latent space visualizations"""
    print("\nCreating visualizations...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Posterior Collapse Analysis
    if results['z_means'] is not None and results['z_logvars'] is not None:
        collapse_stats = analyze_posterior_collapse(results['z_means'], results['z_logvars'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # KL divergence per dimension
        ax = axes[0, 0]
        ax.bar(range(len(collapse_stats['kl_per_dim'])), collapse_stats['kl_per_dim'])
        ax.axhline(y=0.01, color='r', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean KL Divergence')
        ax.set_title('KL Divergence per Latent Dimension')
        ax.legend()
        
        # Variance per dimension
        ax = axes[0, 1]
        ax.bar(range(len(collapse_stats['var_per_dim'])), collapse_stats['var_per_dim'])
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Mean Variance')
        ax.set_title('Variance per Latent Dimension')
        ax.set_yscale('log')
        
        # KL divergence distribution
        ax = axes[1, 0]
        ax.hist(collapse_stats['kl_per_dim'], bins=50, edgecolor='black')
        ax.axvline(x=0.01, color='r', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('KL Divergence')
        ax.set_ylabel('Number of Dimensions')
        ax.set_title('Distribution of KL Divergence across Dimensions')
        ax.legend()
        
        # Active vs Collapsed dimensions pie chart
        ax = axes[1, 1]
        active_dims = len(collapse_stats['kl_per_dim']) - collapse_stats['n_collapsed']
        ax.pie([active_dims, collapse_stats['n_collapsed']], 
               labels=['Active', 'Collapsed'],
               autopct='%1.1f%%',
               colors=['green', 'red'])
        ax.set_title('Active vs Collapsed Latent Dimensions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'posterior_collapse_analysis.png'), dpi=300)
        plt.close()
    
    # 2. Latent Space Visualization (UMAP)
    if results['z_samples'] is not None and results['h_seq'] is not None:
        # Flatten for visualization
        z_flat = results['z_samples'].reshape(-1, results['z_samples'].shape[-1])
        h_flat = results['h_seq'].reshape(-1, results['h_seq'].shape[-1])
        
        # Fit UMAP
        print("Fitting UMAP for latent space...")
        all_latent = np.concatenate([z_flat, h_flat], axis=0)
        
        # Use fewer points if dataset is large
        if len(all_latent) > 5000:
            indices = np.random.choice(len(all_latent), 5000, replace=False)
            all_latent = all_latent[indices]
            n_z = len(indices) // 2
        else:
            n_z = len(z_flat)
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(all_latent)
        
        # Split embeddings
        z_embed = embedding[:n_z]
        h_embed = embedding[n_z:]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Z space (before transformer)
        ax = axes[0]
        scatter = ax.scatter(z_embed[:, 0], z_embed[:, 1], 
                           c=results['labels'][:n_z] if results['labels'] is not None else 'blue',
                           alpha=0.6, s=10, cmap='tab10')
        ax.set_title('Latent Space Z (before transformer)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if results['labels'] is not None:
            plt.colorbar(scatter, ax=ax, label='Class')
        
        # H space (after transformer)
        ax = axes[1]
        scatter = ax.scatter(h_embed[:, 0], h_embed[:, 1],
                           c=results['labels'][:n_z] if results['labels'] is not None else 'green',
                           alpha=0.6, s=10, cmap='tab10')
        ax.set_title('Latent Space H (after transformer)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if results['labels'] is not None:
            plt.colorbar(scatter, ax=ax, label='Class')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_space_umap.png'), dpi=300)
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
            ax.hist(recon_errors, bins=50, edgecolor='black')
            ax.set_xlabel('Mean Squared Error')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Reconstruction Errors')
            ax.axvline(x=np.mean(recon_errors), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(recon_errors):.4f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'reconstruction_errors.png'), dpi=300)
            plt.close()
    
    # 4. Latent Dimension Correlation Heatmap
    if results['z_means'] is not None:
        z_means_flat = results['z_means'].reshape(-1, results['z_means'].shape[-1])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(z_means_flat.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix of Latent Dimensions')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'latent_correlation.png'), dpi=300)
        plt.close()
    
    print(f"\nVisualizations saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize SeqSetVAE model')
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
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--max_batches', type=int, default=10,
                       help='Maximum number of batches to process')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loading')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Load model
    model = load_model(args.checkpoint, config)
    
    # Prepare data
    print("Loading data...")
    data_module = SeqSetVAEDataModule(
        args.data_dir, 
        args.params_map_path, 
        args.label_path,
        batch_size=args.batch_size
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Collect representations
    results = collect_latent_representations(model, test_loader, args.max_batches)
    
    # Create visualizations
    plot_latent_space(results, args.save_dir)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()