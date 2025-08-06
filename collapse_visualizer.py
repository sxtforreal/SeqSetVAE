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

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RealTimeCollapseVisualizer:
    """实时后验塌缩检测可视化工具"""
    
    def __init__(self, log_dir: str, update_interval: int = 1000):
        self.log_dir = log_dir
        self.update_interval = update_interval  # 更新间隔(ms)
        
        # 数据存储
        self.max_points = 1000  # 最多显示的数据点数
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
        
        # 阈值（将从日志文件中读取）
        self.thresholds = {
            'kl_threshold': 0.01,
            'var_threshold': 0.1,
            'active_threshold': 0.1
        }
        
        # 设置图形
        self.setup_figure()
        
        # 状态跟踪
        self.last_modified = 0
        self.log_file_path = None
        
    def setup_figure(self):
        """设置matplotlib图形"""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('VAE后验塌缩实时监控', fontsize=16, fontweight='bold')
        
        # 创建子图
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # KL散度图
        self.ax_kl = self.fig.add_subplot(gs[0, 0])
        self.ax_kl.set_title('KL散度监控')
        self.ax_kl.set_ylabel('KL Divergence')
        self.ax_kl.grid(True, alpha=0.3)
        
        # 方差图
        self.ax_var = self.fig.add_subplot(gs[0, 1])
        self.ax_var.set_title('潜在变量方差监控')
        self.ax_var.set_ylabel('Mean Variance')
        self.ax_var.grid(True, alpha=0.3)
        
        # 激活单元比例图
        self.ax_active = self.fig.add_subplot(gs[1, 0])
        self.ax_active.set_title('激活单元比例监控')
        self.ax_active.set_ylabel('Active Units Ratio')
        self.ax_active.grid(True, alpha=0.3)
        
        # 重构损失图
        self.ax_recon = self.fig.add_subplot(gs[1, 1])
        self.ax_recon.set_title('重构损失监控')
        self.ax_recon.set_ylabel('Reconstruction Loss')
        self.ax_recon.grid(True, alpha=0.3)
        
        # 状态面板
        self.ax_status = self.fig.add_subplot(gs[2, :])
        self.ax_status.set_title('检测状态')
        self.ax_status.axis('off')
        
        # 初始化线条
        self.line_kl, = self.ax_kl.plot([], [], 'b-', linewidth=2, label='KL Divergence')
        self.line_var, = self.ax_var.plot([], [], 'orange', linewidth=2, label='Variance')
        self.line_active, = self.ax_active.plot([], [], 'green', linewidth=2, label='Active Ratio')
        self.line_recon, = self.ax_recon.plot([], [], 'purple', linewidth=2, label='Recon Loss')
        
        # 阈值线（将在数据加载后添加）
        self.threshold_lines = {}
        
    def find_log_file(self):
        """查找最新的日志文件"""
        if not os.path.exists(self.log_dir):
            return None
            
        log_files = [f for f in os.listdir(self.log_dir) if f.startswith('collapse_detection_') and f.endswith('.log')]
        if not log_files:
            return None
            
        # 选择最新的日志文件
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)), reverse=True)
        return os.path.join(self.log_dir, log_files[0])
        
    def parse_log_file(self, log_file: str):
        """解析日志文件获取数据"""
        if not os.path.exists(log_file):
            return False
            
        # 检查文件是否有更新
        current_modified = os.path.getmtime(log_file)
        if current_modified <= self.last_modified:
            return False
            
        self.last_modified = current_modified
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 解析新增的日志行
            new_data_found = False
            
            for line in lines:
                if 'Step' in line and '后验塌缩警告' in line:
                    # 解析警告信息
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
                        
                elif '检测到后验塌缩' in line:
                    # 解析塌缩检测
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
            print(f"解析日志文件失败: {e}")
            return False
            
    def simulate_data(self):
        """模拟数据（用于测试）"""
        if len(self.data['steps']) == 0:
            step = 0
        else:
            step = self.data['steps'][-1] + 1
            
        # 模拟不同阶段的数据
        if step < 200:
            # 正常阶段
            kl = 0.05 + np.random.normal(0, 0.01)
            var = 0.3 + np.random.normal(0, 0.05)
            active = 0.8 + np.random.normal(0, 0.1)
            recon = 2.0 + np.random.normal(0, 0.2)
        elif step < 400:
            # 开始塌缩
            progress = (step - 200) / 200
            kl = 0.05 * (1 - progress) + np.random.normal(0, 0.005)
            var = 0.3 * (1 - progress) + np.random.normal(0, 0.02)
            active = 0.8 * (1 - progress * 0.8) + np.random.normal(0, 0.05)
            recon = 2.0 + progress * 0.5 + np.random.normal(0, 0.1)
        else:
            # 完全塌缩
            kl = 0.001 + np.random.normal(0, 0.0005)
            var = 0.01 + np.random.normal(0, 0.005)
            active = 0.05 + np.random.normal(0, 0.02)
            recon = 2.8 + np.random.normal(0, 0.1)
            
            if not self.data['collapse_detected'] and step > 420:
                self.data['collapse_detected'] = True
                self.data['collapse_step'] = step
                
        # 确保值在合理范围内
        kl = max(0, kl)
        var = max(0, var)
        active = np.clip(active, 0, 1)
        recon = max(0, recon)
        
        self.data['steps'].append(step)
        self.data['kl_divergence'].append(kl)
        self.data['variance'].append(var)
        self.data['active_ratio'].append(active)
        self.data['recon_loss'].append(recon)
        
        # 模拟警告
        if step > 300 and len(self.data['warnings']) < 5 and np.random.random() < 0.1:
            self.data['warnings'].append({
                'step': step,
                'message': f'Step {step}: KL散度过低警告',
                'time': datetime.now()
            })
            
    def update_plot(self, frame):
        """更新图表"""
        # 查找并解析日志文件
        if self.log_file_path is None:
            self.log_file_path = self.find_log_file()
            
        if self.log_file_path:
            self.parse_log_file(self.log_file_path)
        else:
            # 如果没有日志文件，使用模拟数据
            self.simulate_data()
            
        # 如果没有数据，返回
        if len(self.data['steps']) == 0:
            return self.line_kl, self.line_var, self.line_active, self.line_recon
            
        steps = list(self.data['steps'])
        
        # 更新KL散度图
        if self.data['kl_divergence']:
            kl_data = list(self.data['kl_divergence'])
            self.line_kl.set_data(steps, kl_data)
            self.ax_kl.relim()
            self.ax_kl.autoscale_view()
            
            # 添加阈值线
            if 'kl' not in self.threshold_lines:
                self.threshold_lines['kl'] = self.ax_kl.axhline(
                    y=self.thresholds['kl_threshold'], 
                    color='red', linestyle='--', alpha=0.7, label='阈值'
                )
                self.ax_kl.legend()
                
        # 更新方差图
        if self.data['variance']:
            var_data = list(self.data['variance'])
            self.line_var.set_data(steps, var_data)
            self.ax_var.relim()
            self.ax_var.autoscale_view()
            
            if 'var' not in self.threshold_lines:
                self.threshold_lines['var'] = self.ax_var.axhline(
                    y=self.thresholds['var_threshold'],
                    color='red', linestyle='--', alpha=0.7, label='阈值'
                )
                self.ax_var.legend()
                
        # 更新激活单元比例图
        if self.data['active_ratio']:
            active_data = list(self.data['active_ratio'])
            self.line_active.set_data(steps, active_data)
            self.ax_active.relim()
            self.ax_active.autoscale_view()
            
            if 'active' not in self.threshold_lines:
                self.threshold_lines['active'] = self.ax_active.axhline(
                    y=self.thresholds['active_threshold'],
                    color='red', linestyle='--', alpha=0.7, label='阈值'
                )
                self.ax_active.legend()
                
        # 更新重构损失图
        if self.data['recon_loss']:
            recon_data = list(self.data['recon_loss'])
            self.line_recon.set_data(steps, recon_data)
            self.ax_recon.relim()
            self.ax_recon.autoscale_view()
            
        # 更新状态面板
        self.update_status_panel()
        
        return self.line_kl, self.line_var, self.line_active, self.line_recon
        
    def update_status_panel(self):
        """更新状态面板"""
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        # 当前状态
        current_time = datetime.now().strftime("%H:%M:%S")
        status_color = 'red' if self.data['collapse_detected'] else 'green'
        status_text = '后验塌缩' if self.data['collapse_detected'] else '正常'
        
        # 状态信息
        status_info = [
            f"当前时间: {current_time}",
            f"监控状态: {status_text}",
            f"数据点数: {len(self.data['steps'])}",
            f"警告次数: {len(self.data['warnings'])}",
        ]
        
        if self.data['collapse_detected']:
            status_info.append(f"塌缩检测步数: {self.data['collapse_step']}")
            
        # 显示状态信息
        for i, info in enumerate(status_info):
            self.ax_status.text(0.02, 0.8 - i*0.15, info, transform=self.ax_status.transAxes,
                              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor='lightblue', alpha=0.7))
        
        # 显示最近的警告
        if self.data['warnings']:
            recent_warnings = self.data['warnings'][-3:]  # 显示最近3个警告
            warning_text = "最近警告:\n" + "\n".join([
                f"Step {w['step']}: {w['message'].split(':')[-1].strip()}" 
                for w in recent_warnings
            ])
            
            self.ax_status.text(0.5, 0.8, warning_text, transform=self.ax_status.transAxes,
                              fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                              facecolor='yellow', alpha=0.7),
                              verticalalignment='top')
        
        # 添加状态指示器
        indicator_color = 'red' if self.data['collapse_detected'] else 'green'
        circle = plt.Circle((0.95, 0.5), 0.03, color=indicator_color, 
                          transform=self.ax_status.transAxes)
        self.ax_status.add_patch(circle)
        
    def start_monitoring(self):
        """开始监控"""
        print(f"🔍 开始监控后验塌缩...")
        print(f"📁 监控目录: {self.log_dir}")
        print(f"🔄 更新间隔: {self.update_interval}ms")
        print("💡 关闭窗口以停止监控")
        
        # 创建动画
        ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.update_interval,
            blit=False, cache_frame_data=False
        )
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        return ani

def create_dashboard(log_dir: str, update_interval: int = 1000):
    """创建监控面板"""
    visualizer = RealTimeCollapseVisualizer(log_dir, update_interval)
    return visualizer.start_monitoring()

def main():
    parser = argparse.ArgumentParser(description='VAE后验塌缩实时可视化监控')
    
    parser.add_argument('--log_dir', type=str, required=True,
                       help='塌缩检测日志目录')
    parser.add_argument('--update_interval', type=int, default=2000,
                       help='更新间隔(毫秒)')
    parser.add_argument('--demo', action='store_true',
                       help='演示模式（使用模拟数据）')
    
    args = parser.parse_args()
    
    if args.demo:
        print("🎭 演示模式：使用模拟数据")
        args.log_dir = "./demo_logs"
        
    print("🚀 启动VAE后验塌缩实时监控")
    print("=" * 50)
    
    try:
        ani = create_dashboard(args.log_dir, args.update_interval)
        
    except KeyboardInterrupt:
        print("\n⏹️  监控被用户中断")
    except Exception as e:
        print(f"\n❌ 监控过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()