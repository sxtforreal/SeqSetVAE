#!/usr/bin/env python3
"""
Simple JSON Viewer for Recon Eval Results
A lightweight tool to view and analyze reconstruction evaluation JSON files
without requiring external dependencies.
"""

import json
import os
import argparse
from typing import Dict, List, Any
from pathlib import Path

class SimpleReconViewer:
    """Simple viewer for reconstruction evaluation JSON files"""
    
    def __init__(self, json_file_path: str):
        """Initialize viewer with JSON file path"""
        self.json_file_path = json_file_path
        self.data = self._load_json()
        self.summary = self.data.get('summary', {})
        self.details = self.data.get('details', [])
        
    def _load_json(self) -> Dict[str, Any]:
        """Load and parse JSON file"""
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Successfully loaded: {os.path.basename(self.json_file_path)}")
            return data
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return {}
    
    def print_formatted_summary(self):
        """Print a beautifully formatted summary"""
        print("\n" + "🔹" * 25 + " SUMMARY " + "🔹" * 25)
        
        if not self.summary:
            print("❌ No summary data found")
            return
            
        # Basic info
        num_samples = self.summary.get('num_samples', 'N/A')
        print(f"\n📊 **Evaluation Overview**")
        print(f"   Number of samples analyzed: {num_samples}")
        
        # Distance quality assessment
        print(f"\n📏 **Distance Quality** (Lower = Better)")
        l2_mean = self.summary.get('nn_l2_mean', float('inf'))
        l2_median = self.summary.get('nn_l2_median', float('inf'))
        l2_p95 = self.summary.get('nn_l2_p95', float('inf'))
        chamfer = self.summary.get('chamfer_l2_mean', float('inf'))
        
        print(f"   L2 Mean Distance:     {l2_mean:.4f} {self._quality_indicator(l2_mean, [0.2, 0.5, 1.0], reverse=True)}")
        print(f"   L2 Median Distance:   {l2_median:.4f} {self._quality_indicator(l2_median, [0.2, 0.5, 1.0], reverse=True)}")
        print(f"   L2 95th Percentile:   {l2_p95:.4f} {self._quality_indicator(l2_p95, [0.5, 1.0, 2.0], reverse=True)}")
        print(f"   Chamfer L2 Distance:  {chamfer:.4f} {self._quality_indicator(chamfer, [0.2, 0.5, 1.0], reverse=True)}")
        
        # Directional quality
        print(f"\n🧭 **Directional Quality** (Higher = Better)")
        cosine_mean = self.summary.get('dir_cosine_mean', 0)
        print(f"   Cosine Similarity:    {cosine_mean:.4f} {self._quality_indicator(cosine_mean, [0.6, 0.8, 0.9])}")
        
        # Magnitude quality
        print(f"\n📐 **Magnitude Quality**")
        mag_mae = self.summary.get('mag_mae', float('inf'))
        mag_rmse = self.summary.get('mag_rmse', float('inf'))
        mag_corr = self.summary.get('mag_corr', 0)
        scale_ratio = self.summary.get('scale_ratio', 1)
        
        print(f"   Magnitude MAE:        {mag_mae:.4f} {self._quality_indicator(mag_mae, [0.1, 0.3, 0.5], reverse=True)}")
        print(f"   Magnitude RMSE:       {mag_rmse:.4f} {self._quality_indicator(mag_rmse, [0.15, 0.4, 0.7], reverse=True)}")
        print(f"   Magnitude Correlation: {mag_corr:.4f} {self._quality_indicator(mag_corr, [0.6, 0.8, 0.9])}")
        print(f"   Scale Ratio:          {scale_ratio:.4f} {self._scale_quality_indicator(scale_ratio)}")
        
        # Coverage analysis
        print(f"\n🎯 **Coverage Analysis** (Higher = Better)")
        coverage_metrics = {k: v for k, v in self.summary.items() if k.startswith('coverage@')}
        for threshold, coverage in sorted(coverage_metrics.items()):
            th_val = threshold.split('@')[1]
            percentage = coverage * 100
            print(f"   Coverage @ {th_val}:        {coverage:.3f} ({percentage:5.1f}%) {self._quality_indicator(coverage, [0.5, 0.7, 0.9])}")
        
        # Overall assessment
        self._print_overall_assessment()
    
    def _quality_indicator(self, value: float, thresholds: List[float], reverse: bool = False) -> str:
        """Return quality indicator emoji based on thresholds"""
        if value != value:  # NaN check
            return "❓"
        
        if reverse:
            if value <= thresholds[0]:
                return "🟢 Excellent"
            elif value <= thresholds[1]:
                return "🟡 Good"
            elif value <= thresholds[2]:
                return "🟠 Fair"
            else:
                return "🔴 Poor"
        else:
            if value >= thresholds[2]:
                return "🟢 Excellent"
            elif value >= thresholds[1]:
                return "🟡 Good"
            elif value >= thresholds[0]:
                return "🟠 Fair"
            else:
                return "🔴 Poor"
    
    def _scale_quality_indicator(self, scale_ratio: float) -> str:
        """Special quality indicator for scale ratio (should be close to 1.0)"""
        if scale_ratio != scale_ratio:  # NaN check
            return "❓"
        
        deviation = abs(scale_ratio - 1.0)
        if deviation <= 0.05:
            return "🟢 Excellent"
        elif deviation <= 0.15:
            return "🟡 Good"
        elif deviation <= 0.3:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
    
    def _print_overall_assessment(self):
        """Print overall quality assessment"""
        print(f"\n🎯 **Overall Assessment**")
        
        # Calculate overall score
        scores = []
        
        # Distance score (lower is better)
        l2_mean = self.summary.get('nn_l2_mean', float('inf'))
        if l2_mean < 0.2:
            scores.append(4)
        elif l2_mean < 0.5:
            scores.append(3)
        elif l2_mean < 1.0:
            scores.append(2)
        else:
            scores.append(1)
        
        # Cosine score (higher is better)
        cosine_mean = self.summary.get('dir_cosine_mean', 0)
        if cosine_mean > 0.9:
            scores.append(4)
        elif cosine_mean > 0.8:
            scores.append(3)
        elif cosine_mean > 0.6:
            scores.append(2)
        else:
            scores.append(1)
        
        # Coverage score
        coverage_50 = self.summary.get('coverage@0.5', 0)
        if coverage_50 > 0.9:
            scores.append(4)
        elif coverage_50 > 0.7:
            scores.append(3)
        elif coverage_50 > 0.5:
            scores.append(2)
        else:
            scores.append(1)
        
        # Scale score
        scale_ratio = self.summary.get('scale_ratio', 1)
        scale_dev = abs(scale_ratio - 1.0)
        if scale_dev <= 0.05:
            scores.append(4)
        elif scale_dev <= 0.15:
            scores.append(3)
        elif scale_dev <= 0.3:
            scores.append(2)
        else:
            scores.append(1)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score >= 3.5:
            assessment = "🟢 **EXCELLENT** - High quality reconstruction"
        elif avg_score >= 2.5:
            assessment = "🟡 **GOOD** - Satisfactory reconstruction quality"
        elif avg_score >= 1.5:
            assessment = "🟠 **FAIR** - Some reconstruction issues detected"
        else:
            assessment = "🔴 **POOR** - Significant reconstruction problems"
        
        print(f"   {assessment}")
        print(f"   Quality Score: {avg_score:.1f}/4.0")
    
    def print_per_sample_summary(self):
        """Print per-sample analysis summary"""
        if not self.details:
            print("\n❌ No per-sample details available")
            return
        
        print(f"\n📈 **Per-Sample Analysis** ({len(self.details)} samples)")
        print("─" * 60)
        
        # Header
        print(f"{'Sample':<8} {'L2 Mean':<10} {'Cosine':<8} {'Chamfer':<10} {'Mag Corr':<10} {'Sets':<6}")
        print("─" * 60)
        
        for detail in self.details:
            if 'global' in detail:
                sample_idx = detail.get('sample_index', '?')
                global_metrics = detail['global']
                per_set = detail.get('per_set', [])
                
                l2_mean = global_metrics.get('nn_l2_mean', 0)
                cosine = global_metrics.get('dir_cosine_mean', 0)
                chamfer = global_metrics.get('chamfer_l2_mean', 0)
                mag_corr = global_metrics.get('mag_corr', 0)
                
                print(f"{sample_idx:<8} {l2_mean:<10.4f} {cosine:<8.4f} {chamfer:<10.4f} {mag_corr:<10.4f} {len(per_set):<6}")
    
    def analyze_trends(self):
        """Analyze trends across samples"""
        if len(self.details) < 2:
            print("\n❌ Need at least 2 samples for trend analysis")
            return
        
        print(f"\n📊 **Trend Analysis**")
        print("─" * 40)
        
        # Extract time series data
        l2_values = []
        cosine_values = []
        chamfer_values = []
        
        for detail in self.details:
            if 'global' in detail:
                global_metrics = detail['global']
                l2_values.append(global_metrics.get('nn_l2_mean', 0))
                cosine_values.append(global_metrics.get('dir_cosine_mean', 0))
                chamfer_values.append(global_metrics.get('chamfer_l2_mean', 0))
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return "N/A"
            slope = (values[-1] - values[0]) / (len(values) - 1)
            if abs(slope) < 0.001:
                return "Stable ➡️"
            elif slope > 0:
                return f"Increasing ⬆️ (+{slope:.4f})"
            else:
                return f"Decreasing ⬇️ ({slope:.4f})"
        
        print(f"   L2 Distance:      {calculate_trend(l2_values)}")
        print(f"   Cosine Similarity: {calculate_trend(cosine_values)}")
        print(f"   Chamfer Distance: {calculate_trend(chamfer_values)}")
        
        # Stability analysis
        def calculate_stability(values):
            if len(values) < 2:
                return "N/A"
            std_dev = (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            mean_val = sum(values) / len(values)
            cv = std_dev / mean_val if mean_val != 0 else float('inf')
            
            if cv < 0.1:
                return "Very Stable 🟢"
            elif cv < 0.2:
                return "Stable 🟡"
            elif cv < 0.4:
                return "Moderately Stable 🟠"
            else:
                return "Unstable 🔴"
        
        print(f"\n📈 **Stability Analysis**")
        print(f"   L2 Distance:      {calculate_stability(l2_values)}")
        print(f"   Cosine Similarity: {calculate_stability(cosine_values)}")
        print(f"   Chamfer Distance: {calculate_stability(chamfer_values)}")
    
    def export_csv_summary(self, output_path: str):
        """Export summary as CSV for external analysis"""
        try:
            # Create CSV content
            csv_lines = ["Metric,Value"]
            for metric, value in self.summary.items():
                csv_lines.append(f"{metric},{value}")
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(csv_lines))
            
            print(f"📄 CSV summary exported to: {output_path}")
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
    
    def export_detailed_csv(self, output_path: str):
        """Export detailed per-sample data as CSV"""
        if not self.details:
            print("❌ No detailed data to export")
            return
        
        try:
            # Extract all global metrics from all samples
            all_rows = []
            for detail in self.details:
                if 'global' in detail:
                    row = {'sample_index': detail.get('sample_index', 0)}
                    row.update(detail['global'])
                    all_rows.append(row)
            
            if not all_rows:
                print("❌ No valid sample data found")
                return
            
            # Get all unique columns
            all_columns = set()
            for row in all_rows:
                all_columns.update(row.keys())
            all_columns = sorted(list(all_columns))
            
            # Write CSV
            csv_lines = [','.join(all_columns)]
            for row in all_rows:
                values = [str(row.get(col, '')) for col in all_columns]
                csv_lines.append(','.join(values))
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(csv_lines))
            
            print(f"📊 Detailed CSV exported to: {output_path}")
        except Exception as e:
            print(f"❌ Error exporting detailed CSV: {e}")


def find_json_files(directory: str = ".") -> List[str]:
    """Find all recon eval JSON files in directory"""
    json_files = []
    
    search_dirs = [directory]
    if directory == ".":
        search_dirs.extend(["./recon_eval", "./results", "./output"])
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.endswith('.json') and 'reconstruction_eval' in file:
                    json_files.append(os.path.join(search_dir, file))
    
    return json_files


def compare_json_files(json_files: List[str]):
    """Compare multiple JSON files in text format"""
    if len(json_files) < 2:
        print("❌ Need at least 2 files for comparison")
        return
    
    print(f"\n🔄 **Comparing {len(json_files)} Files**")
    print("=" * 80)
    
    # Load all summaries
    summaries = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            filename = os.path.basename(json_file)
            summaries[filename] = data.get('summary', {})
        except Exception as e:
            print(f"❌ Failed to load {json_file}: {e}")
    
    if len(summaries) < 2:
        print("❌ Not enough valid files for comparison")
        return
    
    # Get common metrics
    all_metrics = set()
    for summary in summaries.values():
        all_metrics.update(summary.keys())
    
    # Print comparison table
    print(f"\n📋 **Metrics Comparison Table**")
    print("-" * 100)
    
    # Header
    files = list(summaries.keys())
    header = f"{'Metric':<25}"
    for file in files:
        header += f"{file[:15]:<18}"
    header += "Best"
    print(header)
    print("-" * 100)
    
    # Metrics rows
    key_metrics = ['nn_l2_mean', 'nn_l2_median', 'dir_cosine_mean', 'chamfer_l2_mean', 
                   'mag_corr', 'scale_ratio', 'coverage@0.5', 'coverage@1.0']
    
    for metric in key_metrics:
        if metric in all_metrics:
            row = f"{metric:<25}"
            values = []
            
            for file in files:
                value = summaries[file].get(metric, float('nan'))
                if value != value:  # NaN check
                    row += f"{'N/A':<18}"
                    values.append(None)
                else:
                    row += f"{value:<18.4f}"
                    values.append(value)
            
            # Determine best value
            valid_values = [v for v in values if v is not None]
            if valid_values:
                if metric in ['nn_l2_mean', 'nn_l2_median', 'chamfer_l2_mean', 'mag_mae', 'mag_rmse']:
                    best_val = min(valid_values)
                    best_indicator = "Lower ⬇️"
                elif metric == 'scale_ratio':
                    best_val = min(valid_values, key=lambda x: abs(x - 1.0))
                    best_indicator = "~1.0 🎯"
                else:
                    best_val = max(valid_values)
                    best_indicator = "Higher ⬆️"
                
                best_file_idx = values.index(best_val)
                row += f"{files[best_file_idx][:10]}"
            else:
                row += "N/A"
            
            print(row)
    
    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Simple JSON viewer for recon eval results")
    parser.add_argument("--json_file", type=str, help="Specific JSON file to analyze")
    parser.add_argument("--json_dir", type=str, default=".", help="Directory to search for JSON files")
    parser.add_argument("--compare", action="store_true", help="Compare all found JSON files")
    parser.add_argument("--export_csv", action="store_true", help="Export summary as CSV")
    parser.add_argument("--export_detailed_csv", action="store_true", help="Export detailed data as CSV")
    parser.add_argument("--output_dir", type=str, default="./analysis_output", help="Output directory for exports")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find JSON files
    if args.json_file:
        if os.path.exists(args.json_file):
            json_files = [args.json_file]
        else:
            print(f"❌ File not found: {args.json_file}")
            return
    else:
        json_files = find_json_files(args.json_dir)
    
    if not json_files:
        print("❌ No recon eval JSON files found!")
        print("💡 Make sure you have run: python evaluate_reconstruction.py [args]")
        print("💡 Or specify a file with: --json_file path/to/file.json")
        return
    
    print(f"📁 Found {len(json_files)} JSON file(s):")
    for f in json_files:
        print(f"  • {f}")
    
    # Single file analysis
    if len(json_files) >= 1:
        main_file = json_files[0]
        print(f"\n🎨 **Analyzing: {os.path.basename(main_file)}**")
        
        viewer = SimpleReconViewer(main_file)
        viewer.print_formatted_summary()
        viewer.print_per_sample_summary()
        viewer.analyze_trends()
        
        # Export options
        if args.export_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(args.output_dir, f"summary_{timestamp}.csv")
            viewer.export_csv_summary(csv_path)
        
        if args.export_detailed_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_csv_path = os.path.join(args.output_dir, f"detailed_{timestamp}.csv")
            viewer.export_detailed_csv(detailed_csv_path)
    
    # Multi-file comparison
    if len(json_files) > 1 and args.compare:
        compare_json_files(json_files)
    
    print(f"\n💡 **Next Steps:**")
    print(f"   • For rich visualizations, install dependencies and use recon_eval_visualizer.py")
    print(f"   • Export data as CSV for analysis in Excel/other tools")
    print(f"   • Compare multiple evaluation runs to track improvements")


if __name__ == "__main__":
    main()