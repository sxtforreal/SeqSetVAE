import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
import config

def analyze_posterior_collapse(model, data_loader, num_batches=10):
    """
    分析后验坍缩情况
    """
    model.eval()
    
    all_mus = []
    all_logvars = []
    all_z_samples = []
    all_h_seq = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            var = batch["var"]
            val = batch["val"]
            time = batch["minute"]
            set_ids = batch["set_id"]
            
            sets = model._split_sets(var, val, time, set_ids)
            
            # 收集z和h的统计信息
            S = len(sets)
            z_prims = []
            pos_list = []
            
            for s_dict in sets:
                var, val, time = s_dict["var"], s_dict["val"], s_dict["minute"]
                minute_val = time.unique().float()
                pos_list.append(minute_val)
                
                recon, z_list, _ = model.setvae(var, val)
                z_sample, mu, logvar = z_list[-1]
                
                all_mus.append(mu.cpu().numpy())
                all_logvars.append(logvar.cpu().numpy())
                all_z_samples.append(z_sample.cpu().numpy())
                z_prims.append(z_sample.squeeze(1))
            
            # 获取transformer输出
            z_seq = torch.stack(z_prims, dim=1)
            pos_tensor = torch.stack(pos_list, dim=1)
            z_seq = model._apply_positional_encoding(z_seq, pos_tensor)
            z_seq = torch.nn.functional.layer_norm(z_seq, [z_seq.size(-1)])
            h_seq = model.transformer(z_seq, mask=model._causal_mask(S, z_seq.device))
            
            all_h_seq.append(h_seq.cpu().numpy())
    
    # 分析结果
    all_mus = np.concatenate(all_mus, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)
    all_z_samples = np.concatenate(all_z_samples, axis=0)
    all_h_seq = np.concatenate(all_h_seq, axis=1)
    
    return {
        'mus': all_mus,
        'logvars': all_logvars,
        'z_samples': all_z_samples,
        'h_seq': all_h_seq
    }

def plot_posterior_analysis(stats, save_path=None):
    """
    绘制后验分析图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. μ的分布
    axes[0, 0].hist(stats['mus'].flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title('Distribution of μ (means)')
    axes[0, 0].set_xlabel('μ values')
    axes[0, 0].set_ylabel('Density')
    
    # 2. σ的分布 (exp(0.5 * logvar))
    sigmas = np.exp(0.5 * stats['logvars']).flatten()
    axes[0, 1].hist(sigmas, bins=50, alpha=0.7, density=True)
    axes[0, 1].set_title('Distribution of σ (standard deviations)')
    axes[0, 1].set_xlabel('σ values')
    axes[0, 1].set_ylabel('Density')
    
    # 3. KL散度分布
    kl_divs = 0.5 * np.sum(np.exp(stats['logvars']) + stats['mus']**2 - 1 - stats['logvars'], axis=-1)
    axes[0, 2].hist(kl_divs.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 2].set_title('Distribution of KL Divergences')
    axes[0, 2].set_xlabel('KL Divergence')
    axes[0, 2].set_ylabel('Density')
    
    # 4. Z样本的UMAP
    z_flat = stats['z_samples'].reshape(-1, stats['z_samples'].shape[-1])
    if z_flat.shape[0] > 1000:  # 采样以提高效率
        indices = np.random.choice(z_flat.shape[0], 1000, replace=False)
        z_flat = z_flat[indices]
    
    umap_z = umap.UMAP(n_components=2, random_state=42)
    z_umap = umap_z.fit_transform(z_flat)
    
    scatter = axes[1, 0].scatter(z_umap[:, 0], z_umap[:, 1], 
                                c=np.arange(len(z_umap)), 
                                alpha=0.6, s=20)
    axes[1, 0].set_title('UMAP of Z samples')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    
    # 5. H序列的UMAP
    h_flat = stats['h_seq'].reshape(-1, stats['h_seq'].shape[-1])
    if h_flat.shape[0] > 1000:
        indices = np.random.choice(h_flat.shape[0], 1000, replace=False)
        h_flat = h_flat[indices]
    
    umap_h = umap.UMAP(n_components=2, random_state=42)
    h_umap = umap_h.fit_transform(h_flat)
    
    axes[1, 1].scatter(h_umap[:, 0], h_umap[:, 1], 
                      c=np.arange(len(h_umap)), 
                      alpha=0.6, s=20)
    axes[1, 1].set_title('UMAP of H sequences')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    
    # 6. 后验坍缩指标
    # Active units (方差大于某个阈值的维度比例)
    active_units_ratio = np.mean(np.exp(stats['logvars']) > 0.1, axis=-1)
    axes[1, 2].hist(active_units_ratio.flatten(), bins=30, alpha=0.7, density=True)
    axes[1, 2].set_title('Active Units Ratio')
    axes[1, 2].set_xlabel('Ratio of Active Units')
    axes[1, 2].set_ylabel('Density')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印统计信息
    print("=== Posterior Collapse Analysis ===")
    print(f"Mean μ: {np.mean(stats['mus']):.4f} ± {np.std(stats['mus']):.4f}")
    print(f"Mean σ: {np.mean(sigmas):.4f} ± {np.std(sigmas):.4f}")
    print(f"Mean KL: {np.mean(kl_divs):.4f} ± {np.std(kl_divs):.4f}")
    print(f"Mean Active Units Ratio: {np.mean(active_units_ratio):.4f}")
    
    # 后验坍缩检测
    if np.mean(sigmas) < 0.1:
        print("⚠️  WARNING: Potential posterior collapse detected (low variance)")
    if np.mean(active_units_ratio) < 0.5:
        print("⚠️  WARNING: Many inactive units detected")
    if np.std(stats['mus']) < 0.1:
        print("⚠️  WARNING: Low diversity in means")
    
    return {
        'mean_sigma': np.mean(sigmas),
        'mean_kl': np.mean(kl_divs),
        'active_units_ratio': np.mean(active_units_ratio)
    }

if __name__ == "__main__":
    # 初始化模型
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
        pretrained_ckpt=config.pretrained_ckpt,
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
    )
    
    # 加载训练好的权重（如果存在）
    try:
        pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SeqSetVAE-v2.ckpt"
        ckpt = torch.load(pretrained_ckpt, map_location=torch.device("cpu"))
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Loaded pretrained weights")
    except:
        print("⚠️  No pretrained weights found, using random initialization")
    
    model.eval()
    
    # 数据
    saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # 分析后验坍缩
    print("Analyzing posterior collapse...")
    stats = analyze_posterior_collapse(model, test_loader, num_batches=5)
    
    # 绘制分析图
    metrics = plot_posterior_analysis(stats, save_path="posterior_analysis.png")
    
    print("\n=== Recommendations ===")
    if metrics['mean_sigma'] < 0.1:
        print("- Decrease beta or increase free_bits")
        print("- Add variance regularization")
    if metrics['active_units_ratio'] < 0.5:
        print("- Reduce model capacity or increase regularization")
    if metrics['mean_kl'] < 1.0:
        print("- Increase KL weight gradually during training")