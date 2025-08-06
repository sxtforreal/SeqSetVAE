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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PosteriorCollapseDetector(Callback):
    """
    实时检测VAE后验塌缩的回调类
    
    监控多个关键指标：
    1. KL散度（每层和总体）
    2. 潜在变量的方差
    3. 潜在变量的均值分布
    4. 激活单元数量
    5. 重构误差变化
    """
    
    def __init__(
        self,
        # 检测阈值
        kl_threshold: float = 0.01,          # KL散度过低阈值
        var_threshold: float = 0.1,          # 方差过低阈值  
        active_units_threshold: float = 0.1,  # 激活单元比例阈值
        
        # 监控窗口
        window_size: int = 100,              # 滑动窗口大小
        check_frequency: int = 50,           # 检查频率（每N个step检查一次）
        
        # 早期停止
        early_stop_patience: int = 200,      # 连续多少步检测到塌缩后停止
        auto_save_on_collapse: bool = True,  # 检测到塌缩时自动保存
        
        # 输出设置
        log_dir: str = "./collapse_logs",    # 日志保存目录
        plot_frequency: int = 500,           # 绘图频率
        verbose: bool = True,                # 是否输出详细信息
    ):
        super().__init__()
        
        # 保存参数
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
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化监控变量
        self.reset_monitoring_state()
        
        # 设置文件日志
        self.setup_file_logging()
        
    def reset_monitoring_state(self):
        """重置监控状态"""
        # 历史数据存储（使用deque实现滑动窗口）
        self.kl_history = deque(maxlen=self.window_size)
        self.var_history = deque(maxlen=self.window_size)
        self.mean_history = deque(maxlen=self.window_size)
        self.active_units_history = deque(maxlen=self.window_size)
        self.recon_loss_history = deque(maxlen=self.window_size)
        
        # 塌缩检测状态
        self.collapse_detected = False
        self.collapse_step = None
        self.collapse_consecutive_steps = 0
        self.collapse_warnings = []
        
        # 步数计数
        self.global_step = 0
        
        # 统计信息
        self.collapse_stats = {
            'total_checks': 0,
            'warnings_issued': 0,
            'false_alarms': 0,
        }
        
    def setup_file_logging(self):
        """设置文件日志记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"collapse_detection_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        self.log_file = log_file
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练批次结束后进行检测"""
        self.global_step += 1
        
        # 按频率检查
        if self.global_step % self.check_frequency != 0:
            return
            
        # 提取监控指标
        metrics = self.extract_monitoring_metrics(pl_module, outputs)
        if metrics is None:
            return
            
        # 更新历史数据
        self.update_history(metrics)
        
        # 进行塌缩检测
        collapse_detected, warnings = self.detect_collapse(metrics)
        
        # 处理检测结果
        self.handle_detection_results(trainer, pl_module, collapse_detected, warnings)
        
        # 定期绘图和保存
        if self.global_step % self.plot_frequency == 0:
            self.save_monitoring_plots()
            
    def extract_monitoring_metrics(self, pl_module: LightningModule, outputs) -> Optional[Dict]:
        """从模型中提取监控指标"""
        try:
            metrics = {}
            
            # 1. 提取KL散度（从logged metrics中获取）
            if hasattr(pl_module, 'logged_metrics'):
                logged = pl_module.logged_metrics
                if 'train_kl' in logged:
                    metrics['kl_divergence'] = logged['train_kl'].item()
                if 'train_recon' in logged:
                    metrics['recon_loss'] = logged['train_recon'].item()
                    
            # 2. 从模型中直接提取潜在变量统计信息
            if hasattr(pl_module, 'setvae'):
                # 获取最近一次前向传播的潜在变量
                latent_stats = self.extract_latent_statistics(pl_module)
                if latent_stats:
                    metrics.update(latent_stats)
                    
            return metrics if metrics else None
            
        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
            return None
            
    def extract_latent_statistics(self, pl_module: LightningModule) -> Dict:
        """提取潜在变量的统计信息"""
        stats = {}
        
        try:
            # 运行一个小批次来获取潜在变量统计
            pl_module.eval()
            with torch.no_grad():
                # 这里需要根据您的具体模型结构来调整
                # 假设我们可以通过某种方式获取最近的潜在变量
                
                # 如果模型有存储最近的潜在变量
                if hasattr(pl_module, '_last_z_list') and pl_module._last_z_list:
                    z_list = pl_module._last_z_list
                    
                    # 计算每层的统计信息
                    layer_vars = []
                    layer_means = []
                    active_units_ratios = []
                    
                    for layer_idx, (z_sample, mu, logvar) in enumerate(z_list):
                        # 方差统计
                        var = torch.exp(logvar)
                        mean_var = var.mean().item()
                        layer_vars.append(mean_var)
                        
                        # 均值统计
                        mean_abs_mu = torch.abs(mu).mean().item()
                        layer_means.append(mean_abs_mu)
                        
                        # 激活单元统计（方差大于阈值的单元比例）
                        active_ratio = (var.mean(0) > self.var_threshold).float().mean().item()
                        active_units_ratios.append(active_ratio)
                        
                    stats['layer_variances'] = layer_vars
                    stats['layer_mean_magnitudes'] = layer_means
                    stats['active_units_ratios'] = active_units_ratios
                    
                    # 总体统计
                    stats['mean_variance'] = np.mean(layer_vars)
                    stats['mean_active_ratio'] = np.mean(active_units_ratios)
                    
            pl_module.train()
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to extract latent statistics: {e}")
            return {}
            
    def update_history(self, metrics: Dict):
        """更新历史数据"""
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
        """检测后验塌缩"""
        warnings = []
        collapse_indicators = 0
        
        self.collapse_stats['total_checks'] += 1
        
        # 1. KL散度检测
        if 'kl_divergence' in metrics:
            kl_val = metrics['kl_divergence']
            if kl_val < self.kl_threshold:
                warnings.append(f"KL散度过低: {kl_val:.6f} < {self.kl_threshold}")
                collapse_indicators += 1
                
            # 检查KL散度趋势
            if len(self.kl_history) >= 20:
                recent_kl = list(self.kl_history)[-20:]
                if all(kl < self.kl_threshold for kl in recent_kl):
                    warnings.append("KL散度持续过低（最近20步）")
                    collapse_indicators += 2
                    
        # 2. 方差检测
        if 'mean_variance' in metrics:
            var_val = metrics['mean_variance']
            if var_val < self.var_threshold:
                warnings.append(f"潜在变量方差过低: {var_val:.6f} < {self.var_threshold}")
                collapse_indicators += 1
                
        # 3. 激活单元检测
        if 'mean_active_ratio' in metrics:
            active_ratio = metrics['mean_active_ratio']
            if active_ratio < self.active_units_threshold:
                warnings.append(f"激活单元比例过低: {active_ratio:.3f} < {self.active_units_threshold}")
                collapse_indicators += 1
                
        # 4. 重构损失趋势检测
        if len(self.recon_loss_history) >= 50:
            recent_recon = list(self.recon_loss_history)[-50:]
            recon_trend = np.polyfit(range(len(recent_recon)), recent_recon, 1)[0]
            if recon_trend > 0.001:  # 重构损失持续上升
                warnings.append(f"重构损失持续上升，趋势: {recon_trend:.6f}")
                collapse_indicators += 1
                
        # 综合判断
        collapse_detected = collapse_indicators >= 2  # 至少2个指标异常才判定为塌缩
        
        if warnings:
            self.collapse_stats['warnings_issued'] += 1
            
        return collapse_detected, warnings
        
    def handle_detection_results(self, trainer, pl_module, collapse_detected: bool, warnings: List[str]):
        """处理检测结果"""
        
        if warnings and self.verbose:
            warning_msg = f"Step {self.global_step}: 后验塌缩警告！\n" + "\n".join(f"  - {w}" for w in warnings)
            logger.warning(warning_msg)
            print(f"\n⚠️  {warning_msg}")
            
        if collapse_detected:
            self.collapse_consecutive_steps += 1
            
            if not self.collapse_detected:
                self.collapse_detected = True
                self.collapse_step = self.global_step
                
                collapse_msg = f"🚨 检测到后验塌缩！Step: {self.global_step}"
                logger.error(collapse_msg)
                print(f"\n{collapse_msg}")
                
                # 自动保存模型
                if self.auto_save_on_collapse:
                    self.save_model_on_collapse(trainer, pl_module)
                    
            # 检查是否需要早期停止
            if self.collapse_consecutive_steps >= self.early_stop_patience:
                stop_msg = f"连续{self.early_stop_patience}步检测到后验塌缩，建议停止训练！"
                logger.error(stop_msg)
                print(f"\n🛑 {stop_msg}")
                
                # 这里可以设置trainer.should_stop = True来停止训练
                # 但为了安全起见，我们只记录建议
                
        else:
            # 重置连续塌缩计数
            if self.collapse_consecutive_steps > 0:
                self.collapse_consecutive_steps = 0
                
                if self.collapse_detected:
                    recovery_msg = f"后验塌缩状态恢复，Step: {self.global_step}"
                    logger.info(recovery_msg)
                    print(f"\n✅ {recovery_msg}")
                    
    def save_model_on_collapse(self, trainer, pl_module):
        """在检测到塌缩时保存模型"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.log_dir, f"model_before_collapse_step_{self.global_step}_{timestamp}.ckpt")
            
            trainer.save_checkpoint(save_path)
            logger.info(f"模型已保存到: {save_path}")
            print(f"💾 模型已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            
    def save_monitoring_plots(self):
        """保存监控图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'后验塌缩监控 - Step {self.global_step}', fontsize=16)
            
            # KL散度历史
            if self.kl_history:
                axes[0, 0].plot(list(self.kl_history))
                axes[0, 0].axhline(y=self.kl_threshold, color='r', linestyle='--', alpha=0.7)
                axes[0, 0].set_title('KL散度历史')
                axes[0, 0].set_ylabel('KL Divergence')
                axes[0, 0].grid(True, alpha=0.3)
                
            # 方差历史
            if self.var_history:
                axes[0, 1].plot(list(self.var_history), color='orange')
                axes[0, 1].axhline(y=self.var_threshold, color='r', linestyle='--', alpha=0.7)
                axes[0, 1].set_title('潜在变量方差历史')
                axes[0, 1].set_ylabel('Mean Variance')
                axes[0, 1].grid(True, alpha=0.3)
                
            # 激活单元比例历史
            if self.active_units_history:
                axes[1, 0].plot(list(self.active_units_history), color='green')
                axes[1, 0].axhline(y=self.active_units_threshold, color='r', linestyle='--', alpha=0.7)
                axes[1, 0].set_title('激活单元比例历史')
                axes[1, 0].set_ylabel('Active Units Ratio')
                axes[1, 0].grid(True, alpha=0.3)
                
            # 重构损失历史
            if self.recon_loss_history:
                axes[1, 1].plot(list(self.recon_loss_history), color='purple')
                axes[1, 1].set_title('重构损失历史')
                axes[1, 1].set_ylabel('Reconstruction Loss')
                axes[1, 1].grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.log_dir, f"monitoring_plot_step_{self.global_step}_{timestamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f"📊 监控图表已保存: {plot_path}")
                
        except Exception as e:
            logger.warning(f"保存监控图表失败: {e}")
            
    def on_train_end(self, trainer, pl_module):
        """训练结束时的总结"""
        summary_msg = f"""
        
🎯 后验塌缩检测总结:
===================
总检查次数: {self.collapse_stats['total_checks']}
发出警告次数: {self.collapse_stats['warnings_issued']}
检测到塌缩: {'是' if self.collapse_detected else '否'}
塌缩发生步数: {self.collapse_step if self.collapse_step else 'N/A'}
日志文件: {self.log_file}
        """
        
        logger.info(summary_msg)
        print(summary_msg)
        
        # 保存最终统计信息
        stats_file = os.path.join(self.log_dir, "final_statistics.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(summary_msg)
            f.write(f"\n检测参数:\n")
            f.write(f"KL阈值: {self.kl_threshold}\n")
            f.write(f"方差阈值: {self.var_threshold}\n") 
            f.write(f"激活单元阈值: {self.active_units_threshold}\n")
            f.write(f"检查频率: {self.check_frequency}\n")
            f.write(f"早期停止耐心: {self.early_stop_patience}\n")


# 辅助函数：为现有模型添加潜在变量跟踪
def add_latent_tracking_to_model(model):
    """为模型添加潜在变量跟踪功能"""
    
    original_forward = model.forward
    
    def tracked_forward(self, *args, **kwargs):
        result = original_forward(*args, **kwargs)
        
        # 如果模型有setvae属性，尝试获取潜在变量信息
        if hasattr(self, 'setvae') and hasattr(self.setvae, '_last_z_list'):
            self._last_z_list = self.setvae._last_z_list
            
        return result
        
    # 替换forward方法
    model.forward = tracked_forward.__get__(model, model.__class__)
    
    return model


# 使用示例和测试函数
def test_detector():
    """测试检测器功能"""
    detector = PosteriorCollapseDetector(
        kl_threshold=0.01,
        var_threshold=0.1,
        check_frequency=10,
        verbose=True
    )
    
    print("✅ 后验塌缩检测器创建成功！")
    print(f"📁 日志目录: {detector.log_dir}")
    print(f"🔍 检测参数:")
    print(f"  - KL阈值: {detector.kl_threshold}")
    print(f"  - 方差阈值: {detector.var_threshold}")
    print(f"  - 检查频率: 每{detector.check_frequency}步")
    
    return detector

if __name__ == "__main__":
    test_detector()