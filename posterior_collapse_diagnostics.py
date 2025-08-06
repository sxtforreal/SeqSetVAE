import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns

class PosteriorCollapseDiagnostics:
    """后验坍缩诊断工具"""
    
    def __init__(self, save_dir: str = "./diagnostics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def analyze_kl_divergence(self, z_list: List[Tuple], save_plot: bool = True) -> Dict:
        """分析KL散度分布"""
        kl_stats = {}
        
        for level_idx, (z_sample, mu, logvar) in enumerate(z_list):
            # 计算每个维度的KL散度
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            
            # 统计信息
            kl_mean = kl_per_dim.mean(dim=0)  # 每个维度的平均KL
            kl_std = kl_per_dim.std(dim=0)    # 每个维度的标准差
            
            kl_stats[f'level_{level_idx}'] = {
                'mean_kl_per_dim': kl_mean.cpu().numpy(),
                'std_kl_per_dim': kl_std.cpu().numpy(),
                'total_kl': kl_per_dim.sum().item(),
                'active_dims': (kl_mean > 0.1).sum().item(),
                'collapse_ratio': (kl_mean < 0.01).float().mean().item()
            }
            
            if save_plot:
                self._plot_kl_distribution(kl_mean, level_idx)
                
        return kl_stats
    
    def _plot_kl_distribution(self, kl_per_dim: torch.Tensor, level_idx: int):
        """绘制KL散度分布图"""
        plt.figure(figsize=(12, 4))
        
        # 子图1：KL散度直方图
        plt.subplot(1, 3, 1)
        plt.hist(kl_per_dim.cpu().numpy(), bins=50, alpha=0.7)
        plt.xlabel('KL Divergence per Dimension')
        plt.ylabel('Frequency')
        plt.title(f'Level {level_idx}: KL Distribution')
        plt.axvline(x=0.1, color='r', linestyle='--', label='Active Threshold')
        plt.legend()
        
        # 子图2：维度vs KL值
        plt.subplot(1, 3, 2)
        dims = np.arange(len(kl_per_dim))
        plt.plot(dims, kl_per_dim.cpu().numpy())
        plt.xlabel('Latent Dimension')
        plt.ylabel('KL Divergence')
        plt.title(f'Level {level_idx}: KL per Dimension')
        plt.axhline(y=0.1, color='r', linestyle='--', label='Active Threshold')
        plt.legend()
        
        # 子图3：累积分布
        plt.subplot(1, 3, 3)
        sorted_kl = np.sort(kl_per_dim.cpu().numpy())
        cumulative = np.arange(1, len(sorted_kl) + 1) / len(sorted_kl)
        plt.plot(sorted_kl, cumulative)
        plt.xlabel('KL Divergence')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Level {level_idx}: Cumulative KL')
        plt.axvline(x=0.1, color='r', linestyle='--', label='Active Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'kl_analysis_level_{level_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_latent_space(self, z_samples: torch.Tensor, save_plot: bool = True) -> Dict:
        """分析潜在空间的特性"""
        if z_samples.dim() == 3:  # [batch, seq, dim]
            z_samples = z_samples.view(-1, z_samples.size(-1))  # flatten to [batch*seq, dim]
        
        stats = {}
        
        # 计算各种统计量
        z_mean = z_samples.mean(dim=0)
        z_std = z_samples.std(dim=0)
        z_var = z_samples.var(dim=0)
        
        # 计算相关性矩阵
        correlation_matrix = torch.corrcoef(z_samples.T)
        
        # 计算互信息（简单近似）
        mutual_info_approx = self._estimate_mutual_information(z_samples)
        
        stats.update({
            'mean_activation': z_mean.cpu().numpy(),
            'std_activation': z_std.cpu().numpy(),
            'variance_activation': z_var.cpu().numpy(),
            'correlation_matrix': correlation_matrix.cpu().numpy(),
            'mutual_info_estimate': mutual_info_approx,
            'effective_dims': (z_var > 0.01).sum().item(),
            'total_dims': z_samples.size(-1)
        })
        
        if save_plot:
            self._plot_latent_analysis(stats)
            
        return stats
    
    def _estimate_mutual_information(self, z_samples: torch.Tensor) -> float:
        """简单的互信息估计"""
        # 使用方差作为互信息的粗略估计
        total_variance = z_samples.var(dim=0).sum()
        return total_variance.item()
    
    def _plot_latent_analysis(self, stats: Dict):
        """绘制潜在空间分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1：激活均值和方差
        dims = np.arange(len(stats['mean_activation']))
        axes[0, 0].plot(dims, stats['mean_activation'], label='Mean', alpha=0.7)
        axes[0, 0].plot(dims, stats['std_activation'], label='Std', alpha=0.7)
        axes[0, 0].set_xlabel('Latent Dimension')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Mean and Std per Dimension')
        axes[0, 0].legend()
        
        # 子图2：方差分布
        axes[0, 1].hist(stats['variance_activation'], bins=50, alpha=0.7)
        axes[0, 1].axvline(x=0.01, color='r', linestyle='--', label='Active Threshold')
        axes[0, 1].set_xlabel('Variance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Variance Distribution')
        axes[0, 1].legend()
        
        # 子图3：相关性矩阵热图
        im = axes[1, 0].imshow(stats['correlation_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title('Correlation Matrix')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 子图4：有效维度统计
        labels = ['Effective Dims', 'Collapsed Dims']
        sizes = [stats['effective_dims'], stats['total_dims'] - stats['effective_dims']]
        colors = ['lightgreen', 'lightcoral']
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Effective vs Collapsed Dimensions')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'latent_space_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_training_report(self, model, dataloader, device='cuda') -> Dict:
        """生成训练报告"""
        model.eval()
        all_z_list = []
        all_recon_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # 只分析前10个批次
                    break
                    
                if hasattr(model, 'setvae'):  # SeqSetVAE
                    var, val, time, set_ids, label = (
                        batch["var"].to(device),
                        batch["val"].to(device), 
                        batch["minute"].to(device),
                        batch["set_id"],
                        batch.get("label")
                    )
                    sets = model._split_sets(var, val, time, set_ids)
                    for s_dict in sets:
                        _, z_list, _ = model.setvae(s_dict["var"], s_dict["val"])
                        all_z_list.extend(z_list)
                else:  # SetVAE
                    var, val = batch["var"].to(device), batch["val"].to(device)
                    recon, z_list, _ = model(var, val)
                    all_z_list.extend(z_list)
        
        # 分析KL散度
        kl_stats = self.analyze_kl_divergence(all_z_list)
        
        # 分析潜在空间
        if all_z_list:
            z_samples = torch.cat([z[0] for z in all_z_list], dim=0)
            latent_stats = self.analyze_latent_space(z_samples)
        else:
            latent_stats = {}
        
        # 生成综合报告
        report = {
            'kl_statistics': kl_stats,
            'latent_statistics': latent_stats,
            'summary': self._generate_summary(kl_stats, latent_stats)
        }
        
        # 保存报告
        self._save_report(report)
        
        return report
    
    def _generate_summary(self, kl_stats: Dict, latent_stats: Dict) -> Dict:
        """生成诊断摘要"""
        summary = {}
        
        # KL散度摘要
        total_active_dims = 0
        total_dims = 0
        avg_collapse_ratio = 0
        
        for level, stats in kl_stats.items():
            total_active_dims += stats['active_dims']
            total_dims += len(stats['mean_kl_per_dim'])
            avg_collapse_ratio += stats['collapse_ratio']
        
        if kl_stats:
            avg_collapse_ratio /= len(kl_stats)
        
        summary.update({
            'total_active_dimensions': total_active_dims,
            'total_dimensions': total_dims,
            'average_collapse_ratio': avg_collapse_ratio,
            'posterior_collapse_detected': avg_collapse_ratio > 0.8
        })
        
        # 潜在空间摘要
        if latent_stats:
            summary.update({
                'effective_latent_dims': latent_stats['effective_dims'],
                'latent_utilization_ratio': latent_stats['effective_dims'] / latent_stats['total_dims']
            })
        
        return summary
    
    def _save_report(self, report: Dict):
        """保存诊断报告"""
        import json
        
        # 转换numpy数组为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        report_json = convert_numpy(report)
        
        with open(self.save_dir / 'diagnostic_report.json', 'w') as f:
            json.dump(report_json, f, indent=2)
        
        print(f"诊断报告已保存到: {self.save_dir}")
        print("\n=== 后验坍缩诊断摘要 ===")
        summary = report['summary']
        print(f"总维度数: {summary.get('total_dimensions', 'N/A')}")
        print(f"活跃维度数: {summary.get('total_active_dimensions', 'N/A')}")
        print(f"平均坍缩比例: {summary.get('average_collapse_ratio', 'N/A'):.3f}")
        print(f"检测到后验坍缩: {'是' if summary.get('posterior_collapse_detected', False) else '否'}")
        
        if 'latent_utilization_ratio' in summary:
            print(f"潜在空间利用率: {summary['latent_utilization_ratio']:.3f}")

def main():
    """示例使用方法"""
    # 这里是使用示例，您可以根据需要调整
    print("后验坍缩诊断工具已准备就绪！")
    print("使用方法：")
    print("1. 创建诊断器：diagnostics = PosteriorCollapseDiagnostics()")
    print("2. 生成报告：report = diagnostics.generate_training_report(model, dataloader)")
    print("3. 查看保存的图表和报告文件")

if __name__ == "__main__":
    main()