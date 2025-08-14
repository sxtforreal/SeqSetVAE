import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_label_distribution(label_path):
    """
    分析标签文件中正负样本的分布
    """
    try:
        # 读取标签文件
        label_df = pd.read_csv(label_path)
        print(f"成功读取标签文件: {label_path}")
        print(f"标签文件列名: {label_df.columns.tolist()}")
        print(f"总样本数: {len(label_df)}")
        
        # 查找包含标签的列 (可能是 'in_hospital_mortality', 'label', 'outcome' 等)
        label_columns = [col for col in label_df.columns if 
                        'mortality' in col.lower() or 
                        'label' in col.lower() or 
                        'outcome' in col.lower() or
                        'target' in col.lower()]
        
        if not label_columns:
            print("\n警告：未找到明显的标签列，显示所有列的统计信息：")
            for col in label_df.columns:
                if col not in ['ts_id', 'patient_id', 'id']:
                    print(f"\n列 '{col}' 的分布:")
                    print(label_df[col].value_counts())
        else:
            for label_col in label_columns:
                print(f"\n标签列 '{label_col}' 的分布:")
                
                # 计算正负样本数量
                value_counts = label_df[label_col].value_counts().sort_index()
                print(value_counts)
                
                # 计算比例
                total = len(label_df)
                print(f"\n比例分析:")
                for value, count in value_counts.items():
                    percentage = (count / total) * 100
                    print(f"  标签 {value}: {count} 样本 ({percentage:.2f}%)")
                
                # 计算不平衡比例
                if len(value_counts) == 2:
                    values = value_counts.values
                    imbalance_ratio = max(values) / min(values)
                    print(f"\n不平衡比例: {imbalance_ratio:.2f}:1")
                    
                    # 判断正负标签
                    if 0 in value_counts and 1 in value_counts:
                        neg_count = value_counts[0]
                        pos_count = value_counts[1]
                        print(f"\n负样本(0): {neg_count} ({neg_count/total*100:.2f}%)")
                        print(f"正样本(1): {pos_count} ({pos_count/total*100:.2f}%)")
                        print(f"正负样本比例: 1:{neg_count/pos_count:.2f}")
        
        # 按数据集划分统计（如果有相关信息）
        if 'partition' in label_df.columns or 'split' in label_df.columns:
            split_col = 'partition' if 'partition' in label_df.columns else 'split'
            print(f"\n按{split_col}划分的标签分布:")
            for split in label_df[split_col].unique():
                split_data = label_df[label_df[split_col] == split]
                print(f"\n{split}集:")
                for label_col in label_columns:
                    counts = split_data[label_col].value_counts().sort_index()
                    for value, count in counts.items():
                        print(f"  标签 {value}: {count} ({count/len(split_data)*100:.2f}%)")
                        
    except FileNotFoundError:
        print(f"错误：找不到文件 {label_path}")
        print("请确保提供正确的标签文件路径")
    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 默认路径
    default_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    
    # 如果提供了命令行参数，使用提供的路径
    if len(sys.argv) > 1:
        label_path = sys.argv[1]
    else:
        label_path = default_path
        
    # 如果默认路径不存在，尝试一些常见的路径
    if not Path(label_path).exists():
        possible_paths = [
            "oc.csv",
            "labels.csv", 
            "label.csv",
            "outcomes.csv",
            "mortality.csv",
            "../oc.csv",
            "../labels.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                label_path = path
                print(f"找到标签文件: {path}")
                break
        else:
            print(f"请提供正确的标签文件路径作为命令行参数")
            print(f"用法: python {sys.argv[0]} <标签文件路径>")
            sys.exit(1)
    
    analyze_label_distribution(label_path)