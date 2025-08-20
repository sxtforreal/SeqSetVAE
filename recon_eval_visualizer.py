#!/usr/bin/env python3
"""
Recon Eval JSON Visualizer
A comprehensive tool to visualize reconstruction evaluation results from JSON files.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ReconEvalVisualizer:
    """Visualizer for reconstruction evaluation JSON files"""
    
    def __init__(self, json_file_path: str):
        """Initialize visualizer with JSON file path"""
        self.json_file_path = json_file_path
        self.data = self._load_json()
        self.summary = self.data.get('summary', {})
        self.details = self.data.get('details', [])
        
    def _load_json(self) -> Dict[str, Any]:
        """Load and parse JSON file"""
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Successfully loaded JSON file: {self.json_file_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return {}
    
    def print_summary(self):
        """Print a formatted summary of the evaluation results"""
        print("\n" + "="*60)
        print("🔍 RECONSTRUCTION EVALUATION SUMMARY")
        print("="*60)
        
        if not self.summary:
            print("❌ No summary data found")
            return
            
        print(f"📊 Number of samples: {self.summary.get('num_samples', 'N/A')}")
        print("\n📏 Distance Metrics:")
        print(f"  • L2 Mean:      {self.summary.get('nn_l2_mean', 'N/A'):.4f}")
        print(f"  • L2 Median:    {self.summary.get('nn_l2_median', 'N/A'):.4f}")
        print(f"  • L2 95th %:    {self.summary.get('nn_l2_p95', 'N/A'):.4f}")
        print(f"  • Chamfer L2:   {self.summary.get('chamfer_l2_mean', 'N/A'):.4f}")
        
        print("\n🧭 Directional Similarity:")
        print(f"  • Cosine Mean:  {self.summary.get('dir_cosine_mean', 'N/A'):.4f}")
        
        print("\n📐 Magnitude Metrics:")
        print(f"  • MAE:          {self.summary.get('mag_mae', 'N/A'):.4f}")
        print(f"  • RMSE:         {self.summary.get('mag_rmse', 'N/A'):.4f}")
        print(f"  • Correlation:  {self.summary.get('mag_corr', 'N/A'):.4f}")
        print(f"  • Scale Ratio:  {self.summary.get('scale_ratio', 'N/A'):.4f}")
        
        # Coverage metrics
        coverage_metrics = {k: v for k, v in self.summary.items() if k.startswith('coverage@')}
        if coverage_metrics:
            print("\n🎯 Coverage at Thresholds:")
            for threshold, coverage in coverage_metrics.items():
                th_val = threshold.split('@')[1]
                print(f"  • @ {th_val}:        {coverage:.3f} ({coverage*100:.1f}%)")
    
    def create_summary_dashboard(self, save_path: Optional[str] = None, show: bool = True):
        """Create a comprehensive dashboard visualization"""
        if not self.summary:
            print("❌ No summary data to visualize")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Reconstruction Evaluation Dashboard\n{os.path.basename(self.json_file_path)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Distance Metrics Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        distance_metrics = ['nn_l2_mean', 'nn_l2_median', 'nn_l2_p95', 'chamfer_l2_mean']
        distance_values = [self.summary.get(m, 0) for m in distance_metrics]
        distance_labels = ['L2 Mean', 'L2 Median', 'L2 95th%', 'Chamfer L2']
        
        bars = ax1.bar(distance_labels, distance_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Distance Metrics', fontweight='bold')
        ax1.set_ylabel('Distance')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, distance_values):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distance_values)*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Coverage Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        coverage_data = []
        thresholds = []
        for k, v in self.summary.items():
            if k.startswith('coverage@'):
                threshold = float(k.split('@')[1])
                thresholds.append(threshold)
                coverage_data.append(v * 100)  # Convert to percentage
        
        if coverage_data:
            ax2.plot(thresholds, coverage_data, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
            ax2.fill_between(thresholds, coverage_data, alpha=0.3, color='#FF6B6B')
            ax2.set_title('Coverage at Distance Thresholds', fontweight='bold')
            ax2.set_xlabel('Distance Threshold')
            ax2.set_ylabel('Coverage (%)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
        
        # 3. Magnitude Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        mag_metrics = ['mag_mae', 'mag_rmse', 'mag_corr']
        mag_values = [self.summary.get(m, 0) for m in mag_metrics]
        mag_labels = ['MAE', 'RMSE', 'Correlation']
        
        # Use different colors for correlation (should be close to 1)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax3.bar(mag_labels, mag_values, color=colors)
        ax3.set_title('Magnitude Metrics', fontweight='bold')
        ax3.set_ylabel('Value')
        
        # Add value labels
        for bar, val in zip(bars, mag_values):
            if val != 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mag_values)*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Per-Sample Analysis (if details available)
        if self.details:
            ax4 = fig.add_subplot(gs[1, :])
            self._plot_per_sample_metrics(ax4)
        
        # 5. Metric Radar Chart
        ax5 = fig.add_subplot(gs[2, 0], projection='polar')
        self._create_radar_chart(ax5)
        
        # 6. Quality Assessment
        ax6 = fig.add_subplot(gs[2, 1])
        self._create_quality_assessment(ax6)
        
        # 7. Scale Analysis
        ax7 = fig.add_subplot(gs[2, 2])
        self._create_scale_analysis(ax7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Dashboard saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_per_sample_metrics(self, ax):
        """Plot per-sample metrics analysis"""
        if not self.details:
            ax.text(0.5, 0.5, 'No per-sample data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Per-Sample Analysis', fontweight='bold')
            return
        
        # Extract metrics for each sample
        sample_indices = []
        l2_means = []
        cosine_means = []
        chamfer_scores = []
        
        for detail in self.details:
            if 'global' in detail:
                sample_indices.append(detail.get('sample_index', len(sample_indices)))
                global_metrics = detail['global']
                l2_means.append(global_metrics.get('nn_l2_mean', 0))
                cosine_means.append(global_metrics.get('dir_cosine_mean', 0))
                chamfer_scores.append(global_metrics.get('chamfer_l2_mean', 0))
        
        if sample_indices:
            x = np.arange(len(sample_indices))
            width = 0.25
            
            ax.bar(x - width, l2_means, width, label='L2 Mean', alpha=0.8, color='#FF6B6B')
            ax.bar(x, cosine_means, width, label='Cosine Mean', alpha=0.8, color='#4ECDC4')
            ax.bar(x + width, chamfer_scores, width, label='Chamfer L2', alpha=0.8, color='#45B7D1')
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Metric Value')
            ax.set_title('Per-Sample Reconstruction Quality', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'S{i}' for i in sample_indices])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _create_radar_chart(self, ax):
        """Create a radar chart for normalized metrics"""
        # Define metrics for radar chart (normalize to 0-1 scale)
        metrics = {
            'Cosine Sim': self.summary.get('dir_cosine_mean', 0),  # Already 0-1
            'Coverage@0.5': self.summary.get('coverage@0.5', 0),  # Already 0-1
            'Coverage@1.0': self.summary.get('coverage@1.0', 0),  # Already 0-1
            'Mag Corr': max(0, self.summary.get('mag_corr', 0)),  # Clamp to 0-1
            'Scale Ratio': min(1, self.summary.get('scale_ratio', 1)),  # Normalize around 1
        }
        
        # Handle NaN values
        for k, v in metrics.items():
            if np.isnan(v) or np.isinf(v):
                metrics[k] = 0
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Add first point to close the radar chart
        values += values[:1]
        
        # Compute angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Quality Radar\n(Higher = Better)', fontweight='bold', pad=20)
        ax.grid(True)
    
    def _create_quality_assessment(self, ax):
        """Create overall quality assessment visualization"""
        # Define quality thresholds and scoring
        l2_mean = self.summary.get('nn_l2_mean', float('inf'))
        cosine_mean = self.summary.get('dir_cosine_mean', 0)
        coverage_50 = self.summary.get('coverage@0.5', 0)
        mag_corr = self.summary.get('mag_corr', 0)
        
        # Quality scoring (0-100)
        scores = []
        labels = []
        
        # L2 Distance (lower is better, score inversely)
        l2_score = max(0, 100 * (1 - min(l2_mean / 2.0, 1)))  # Assume 2.0 is poor
        scores.append(l2_score)
        labels.append('Distance\nQuality')
        
        # Cosine similarity (higher is better)
        cosine_score = max(0, cosine_mean * 100)
        scores.append(cosine_score)
        labels.append('Direction\nQuality')
        
        # Coverage (higher is better)
        coverage_score = coverage_50 * 100
        scores.append(coverage_score)
        labels.append('Coverage\nQuality')
        
        # Magnitude correlation (higher is better, handle NaN)
        if np.isnan(mag_corr) or np.isinf(mag_corr):
            mag_score = 0
        else:
            mag_score = max(0, mag_corr * 100)
        scores.append(mag_score)
        labels.append('Magnitude\nQuality')
        
        # Create horizontal bar chart
        colors = ['#FF6B6B' if s < 50 else '#4ECDC4' if s < 80 else '#96CEB4' for s in scores]
        bars = ax.barh(labels, scores, color=colors)
        
        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}%', va='center', fontweight='bold')
        
        ax.set_xlim(0, 105)
        ax.set_xlabel('Quality Score (%)')
        ax.set_title('Quality Assessment', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _create_scale_analysis(self, ax):
        """Analyze scale ratio and magnitude statistics"""
        scale_ratio = self.summary.get('scale_ratio', 1)
        orig_norm_mean = self.summary.get('orig_norm_mean', 0)
        recon_norm_mean = self.summary.get('recon_norm_mean', 0)
        orig_norm_std = self.summary.get('orig_norm_std', 0)
        recon_norm_std = self.summary.get('recon_norm_std', 0)
        
        # Create comparison bars
        categories = ['Mean Norm', 'Std Norm']
        orig_values = [orig_norm_mean, orig_norm_std]
        recon_values = [recon_norm_mean, recon_norm_std]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, orig_values, width, label='Original', color='#4ECDC4', alpha=0.8)
        ax.bar(x + width/2, recon_values, width, label='Reconstructed', color='#FF6B6B', alpha=0.8)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(f'Scale Analysis\n(Ratio: {scale_ratio:.3f})', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_detailed_analysis(self, save_path: Optional[str] = None, show: bool = True):
        """Create detailed per-sample analysis"""
        if not self.details:
            print("❌ No detailed data available for analysis")
            return
        
        # Extract per-sample data
        sample_data = []
        for detail in self.details:
            if 'global' in detail:
                sample_info = {
                    'sample_index': detail.get('sample_index', 0),
                    **detail['global']
                }
                sample_data.append(sample_info)
        
        if not sample_data:
            print("❌ No valid sample data found")
            return
        
        df = pd.DataFrame(sample_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Per-Sample Analysis', fontsize=16, fontweight='bold')
        
        # 1. L2 Distance Evolution
        if 'nn_l2_mean' in df.columns:
            axes[0, 0].plot(df['sample_index'], df['nn_l2_mean'], 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_title('L2 Distance Across Samples')
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel('L2 Mean Distance')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cosine Similarity Evolution
        if 'dir_cosine_mean' in df.columns:
            axes[0, 1].plot(df['sample_index'], df['dir_cosine_mean'], 'o-', 
                          linewidth=2, markersize=8, color='#4ECDC4')
            axes[0, 1].set_title('Directional Cosine Similarity')
            axes[0, 1].set_xlabel('Sample Index')
            axes[0, 1].set_ylabel('Cosine Similarity')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Coverage Heatmap
        coverage_cols = [col for col in df.columns if col.startswith('coverage@')]
        if coverage_cols:
            coverage_matrix = df[coverage_cols].values
            im = axes[0, 2].imshow(coverage_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            axes[0, 2].set_title('Coverage Heatmap')
            axes[0, 2].set_xlabel('Sample Index')
            axes[0, 2].set_ylabel('Distance Threshold')
            axes[0, 2].set_yticks(range(len(coverage_cols)))
            axes[0, 2].set_yticklabels([col.split('@')[1] for col in coverage_cols])
            plt.colorbar(im, ax=axes[0, 2], label='Coverage')
        
        # 4. Magnitude Correlation
        if 'mag_corr' in df.columns:
            valid_corr = df['mag_corr'].dropna()
            if len(valid_corr) > 0:
                axes[1, 0].bar(range(len(valid_corr)), valid_corr, color='#96CEB4')
                axes[1, 0].set_title('Magnitude Correlation per Sample')
                axes[1, 0].set_xlabel('Sample Index')
                axes[1, 0].set_ylabel('Correlation')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
                axes[1, 0].set_ylim(-1, 1)
        
        # 5. Scale Ratio Analysis
        if 'scale_ratio' in df.columns:
            axes[1, 1].plot(df['sample_index'], df['scale_ratio'], 'o-', 
                          linewidth=2, markersize=8, color='#FFD93D')
            axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Scale')
            axes[1, 1].set_title('Scale Ratio Across Samples')
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Scale Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Chamfer Distance
        if 'chamfer_l2_mean' in df.columns:
            axes[1, 2].plot(df['sample_index'], df['chamfer_l2_mean'], 'o-', 
                          linewidth=2, markersize=8, color='#B19CD9')
            axes[1, 2].set_title('Chamfer Distance')
            axes[1, 2].set_xlabel('Sample Index')
            axes[1, 2].set_ylabel('Chamfer L2 Mean')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            detailed_path = save_path.replace('.png', '_detailed.png')
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            print(f"💾 Detailed analysis saved to: {detailed_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_comparison_table(self):
        """Create a formatted comparison table"""
        print("\n" + "="*80)
        print("📋 DETAILED METRICS TABLE")
        print("="*80)
        
        # Summary metrics table
        summary_df = pd.DataFrame([self.summary]).T
        summary_df.columns = ['Value']
        summary_df.index.name = 'Metric'
        
        print("\n🔍 Summary Metrics:")
        print(summary_df.to_string(float_format='%.4f'))
        
        # Per-sample table if available
        if self.details:
            print(f"\n📊 Per-Sample Details ({len(self.details)} samples):")
            sample_rows = []
            for detail in self.details:
                if 'global' in detail:
                    row = {'Sample': detail.get('sample_index', 'N/A')}
                    row.update(detail['global'])
                    sample_rows.append(row)
            
            if sample_rows:
                sample_df = pd.DataFrame(sample_rows)
                # Select key columns for display
                key_cols = ['Sample', 'nn_l2_mean', 'dir_cosine_mean', 'chamfer_l2_mean', 
                           'mag_corr', 'scale_ratio', 'coverage@0.5']
                display_cols = [col for col in key_cols if col in sample_df.columns]
                print(sample_df[display_cols].to_string(index=False, float_format='%.4f'))
    
    def export_summary_report(self, output_path: str):
        """Export a comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Reconstruction Evaluation Report
Generated: {timestamp}
Source: {self.json_file_path}

## Summary Metrics
"""
        
        for metric, value in self.summary.items():
            if isinstance(value, (int, float)):
                report += f"- **{metric}**: {value:.4f}\n"
            else:
                report += f"- **{metric}**: {value}\n"
        
        if self.details:
            report += f"\n## Sample Details\nEvaluated {len(self.details)} samples:\n\n"
            for i, detail in enumerate(self.details):
                if 'global' in detail:
                    global_metrics = detail['global']
                    report += f"### Sample {detail.get('sample_index', i)}\n"
                    for k, v in global_metrics.items():
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            report += f"- {k}: {v:.4f}\n"
                    report += "\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Summary report exported to: {output_path}")


def compare_multiple_files(json_files: List[str], save_path: Optional[str] = None, show: bool = True):
    """Compare metrics across multiple JSON files"""
    if len(json_files) < 2:
        print("❌ Need at least 2 JSON files for comparison")
        return
    
    # Load all files
    all_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            filename = os.path.basename(json_file)
            all_data[filename] = data.get('summary', {})
            print(f"✅ Loaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to load {json_file}: {e}")
    
    if len(all_data) < 2:
        print("❌ Not enough valid files loaded")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_data).T
    comparison_df = comparison_df.select_dtypes(include=[np.number])  # Only numeric columns
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-File Comparison Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Key metrics comparison
    key_metrics = ['nn_l2_mean', 'dir_cosine_mean', 'chamfer_l2_mean', 'mag_corr']
    available_metrics = [m for m in key_metrics if m in comparison_df.columns]
    
    if available_metrics:
        comparison_df[available_metrics].plot(kind='bar', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title('Key Metrics Comparison')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Coverage comparison
    coverage_cols = [col for col in comparison_df.columns if col.startswith('coverage@')]
    if coverage_cols:
        comparison_df[coverage_cols].plot(kind='line', marker='o', ax=axes[0, 1])
        axes[0, 1].set_title('Coverage Comparison')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Correlation heatmap
    if len(comparison_df) > 1:
        corr_matrix = comparison_df.T.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], square=True, fmt='.3f')
        axes[1, 0].set_title('File Correlation Matrix')
    
    # 4. Ranking comparison
    ranking_metrics = ['nn_l2_mean', 'dir_cosine_mean', 'coverage@0.5']
    available_ranking = [m for m in ranking_metrics if m in comparison_df.columns]
    
    if available_ranking:
        # Rank files (lower is better for distance, higher for others)
        ranks = pd.DataFrame(index=comparison_df.index)
        for metric in available_ranking:
            if 'l2' in metric or 'chamfer' in metric:
                ranks[metric] = comparison_df[metric].rank(ascending=True)
            else:
                ranks[metric] = comparison_df[metric].rank(ascending=False)
        
        ranks.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Ranking Comparison\n(Lower rank = Better)')
        axes[1, 1].set_ylabel('Rank')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize recon eval JSON results")
    parser.add_argument("--json_file", type=str, help="Path to JSON file to visualize")
    parser.add_argument("--json_dir", type=str, help="Directory containing JSON files")
    parser.add_argument("--compare", action="store_true", help="Compare multiple JSON files")
    parser.add_argument("--save_dir", type=str, default="./visualizations", 
                       help="Directory to save visualizations")
    parser.add_argument("--export_report", action="store_true", 
                       help="Export markdown summary report")
    parser.add_argument("--no_show", action="store_true", help="Don't display plots")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Find JSON files
    json_files = []
    
    if args.json_file:
        if os.path.exists(args.json_file):
            json_files.append(args.json_file)
        else:
            print(f"❌ File not found: {args.json_file}")
            return
    
    if args.json_dir:
        if os.path.exists(args.json_dir):
            for file in os.listdir(args.json_dir):
                if file.endswith('.json') and 'reconstruction_eval' in file:
                    json_files.append(os.path.join(args.json_dir, file))
        else:
            print(f"❌ Directory not found: {args.json_dir}")
            return
    
    # Auto-discover if no files specified
    if not json_files:
        print("🔍 Auto-discovering JSON files...")
        search_dirs = ['./recon_eval', './']
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.json') and 'reconstruction_eval' in file:
                        json_files.append(os.path.join(search_dir, file))
    
    if not json_files:
        print("❌ No recon eval JSON files found!")
        print("💡 Try running: python evaluate_reconstruction.py --help")
        return
    
    print(f"📁 Found {len(json_files)} JSON file(s):")
    for f in json_files:
        print(f"  • {f}")
    
    # Single file visualization
    if len(json_files) == 1 or not args.compare:
        json_file = json_files[0]
        print(f"\n🎨 Visualizing: {json_file}")
        
        visualizer = ReconEvalVisualizer(json_file)
        
        # Print summary
        visualizer.print_summary()
        visualizer.create_comparison_table()
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        
        dashboard_path = os.path.join(args.save_dir, f"{base_name}_dashboard_{timestamp}.png")
        detailed_path = os.path.join(args.save_dir, f"{base_name}_detailed_{timestamp}.png")
        
        visualizer.create_summary_dashboard(dashboard_path, show=not args.no_show)
        visualizer.create_detailed_analysis(detailed_path, show=not args.no_show)
        
        # Export report if requested
        if args.export_report:
            report_path = os.path.join(args.save_dir, f"{base_name}_report_{timestamp}.md")
            visualizer.export_summary_report(report_path)
    
    # Multi-file comparison
    if len(json_files) > 1 and args.compare:
        print(f"\n🔄 Comparing {len(json_files)} files...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(args.save_dir, f"comparison_{timestamp}.png")
        compare_multiple_files(json_files, comparison_path, show=not args.no_show)


if __name__ == "__main__":
    main()