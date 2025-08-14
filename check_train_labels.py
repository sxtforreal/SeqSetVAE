import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SeqSetVAEDataModule
import pandas as pd
import numpy as np

def analyze_dataset_labels():
    """直接从数据集模块分析标签分布"""
    
    # 使用你的实际路径
    data_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    
    print("正在加载数据模块...")
    
    try:
        # 创建数据模块
        data_module = SeqSetVAEDataModule(
            saved_dir=data_dir,
            params_map_path=params_map_path,
            label_path=label_path,
            batch_size=1
        )
        
        # 设置数据集
        data_module.setup()
        
        print("\n=== 数据集大小 ===")
        print(f"训练集: {len(data_module.train_dataset)} 个样本")
        print(f"验证集: {len(data_module.val_dataset)} 个样本")
        print(f"测试集: {len(data_module.test_dataset)} 个样本")
        
        # 分析每个数据集的标签分布
        datasets = {
            'train': data_module.train_dataset,
            'valid': data_module.val_dataset,
            'test': data_module.test_dataset
        }
        
        overall_stats = {}
        
        for name, dataset in datasets.items():
            print(f"\n=== {name.upper()} 数据集标签分布 ===")
            
            labels = []
            missing_labels = 0
            
            # 遍历所有患者ID并查找标签
            for patient_id in dataset.patient_ids:
                try:
                    # 尝试不同的ID转换方式
                    pid = int(float(patient_id))
                except:
                    try:
                        pid = int(patient_id)
                    except:
                        pid = patient_id
                
                # 查找标签
                if pid in data_module.label_map:
                    labels.append(data_module.label_map[pid])
                else:
                    # 尝试其他形式
                    found = False
                    for key in data_module.label_map.keys():
                        if str(key) == str(patient_id) or str(key) == str(pid):
                            labels.append(data_module.label_map[key])
                            found = True
                            break
                    if not found:
                        missing_labels += 1
                        labels.append(0)  # 默认标签
            
            # 统计
            labels = np.array(labels)
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            print(f"总样本数: {len(labels)}")
            if missing_labels > 0:
                print(f"缺失标签数: {missing_labels}")
            
            for label, count in zip(unique_labels, counts):
                percentage = (count / len(labels)) * 100
                print(f"  标签 {label}: {count} 样本 ({percentage:.2f}%)")
            
            # 计算正负样本比例
            if 0 in unique_labels and 1 in unique_labels:
                neg_idx = np.where(unique_labels == 0)[0][0]
                pos_idx = np.where(unique_labels == 1)[0][0]
                neg_count = counts[neg_idx]
                pos_count = counts[pos_idx]
                
                print(f"\n负样本(0): {neg_count} ({neg_count/len(labels)*100:.2f}%)")
                print(f"正样本(1): {pos_count} ({pos_count/len(labels)*100:.2f}%)")
                print(f"正负样本比例: 1:{neg_count/pos_count:.2f}")
                print(f"类别不平衡比例: {neg_count/pos_count:.2f}:1")
                
                overall_stats[name] = {
                    'total': len(labels),
                    'positive': pos_count,
                    'negative': neg_count,
                    'positive_rate': pos_count/len(labels),
                    'imbalance_ratio': neg_count/pos_count
                }
        
        # 总体统计
        print("\n=== 总体统计 ===")
        total_samples = sum(stats['total'] for stats in overall_stats.values())
        total_positive = sum(stats['positive'] for stats in overall_stats.values())
        total_negative = sum(stats['negative'] for stats in overall_stats.values())
        
        print(f"总样本数: {total_samples}")
        print(f"总正样本数: {total_positive} ({total_positive/total_samples*100:.2f}%)")
        print(f"总负样本数: {total_negative} ({total_negative/total_samples*100:.2f}%)")
        print(f"总体正负样本比例: 1:{total_negative/total_positive:.2f}")
        
        # 给出建议
        print("\n=== 建议 ===")
        imbalance_ratio = total_negative / total_positive
        
        if imbalance_ratio > 10:
            print("⚠️ 严重的类别不平衡！")
            print("建议措施：")
            print("1. 使用 Focal Loss 或 Class-Balanced Loss")
            print("2. 使用 SMOTE 或其他过采样技术")
            print("3. 使用加权采样器 (WeightedRandomSampler)")
            print("4. 调整正样本的损失权重为:", f"{imbalance_ratio:.2f}")
            print("5. 使用 AUPRC 而不是 AUC 作为主要评估指标")
        elif imbalance_ratio > 5:
            print("⚠️ 中度类别不平衡")
            print("建议使用类别权重或 Focal Loss")
        else:
            print("✓ 类别分布相对平衡")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_dataset_labels()