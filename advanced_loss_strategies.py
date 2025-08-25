"""
Advanced Loss Function Strategies for Medical Classification
多种损失函数组合策略，用于提升AUC和AUPRC性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import FocalLoss


class DiversifiedLossStrategy:
    """
    多样化损失策略类
    主要目标：通过不同的损失函数组合，让主辅助头学习互补的表示
    """
    
    def __init__(self, strategy="balanced", focal_alpha=0.2, focal_gamma=2.5):
        self.strategy = strategy
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        
    def compute_loss(self, main_logits, aux_logits, labels):
        """
        计算多样化损失
        
        Args:
            main_logits: 主预测头的logits [B, num_classes]
            aux_logits: 辅助预测头的logits [B, num_classes] 
            labels: 真实标签 [B]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各组件损失的字典
        """
        
        if self.strategy == "balanced":
            return self._balanced_strategy(main_logits, aux_logits, labels)
        elif self.strategy == "focal_smooth":
            return self._focal_smooth_strategy(main_logits, aux_logits, labels)
        elif self.strategy == "confidence_aware":
            return self._confidence_aware_strategy(main_logits, aux_logits, labels)
        elif self.strategy == "adversarial":
            return self._adversarial_strategy(main_logits, aux_logits, labels)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _balanced_strategy(self, main_logits, aux_logits, labels):
        """
        平衡策略：主头用Focal Loss，辅助头用平滑交叉熵
        适用于大多数不平衡分类任务
        """
        # 主损失：Focal Loss (处理类别不平衡和困难样本)
        main_loss = self.focal_loss(main_logits, labels)
        
        # 辅助损失：标签平滑交叉熵 (更平滑的概率分布)
        aux_loss = F.cross_entropy(aux_logits, labels, label_smoothing=0.15)
        
        # 一致性损失：鼓励两个头学习相似但不完全相同的表示
        main_probs = F.softmax(main_logits, dim=1)
        consistency_loss = F.kl_div(
            F.log_softmax(aux_logits, dim=1),
            main_probs.detach(),
            reduction='batchmean'
        )
        
        # 组合损失 (70% + 25% + 5%)
        total_loss = 0.7 * main_loss + 0.25 * aux_loss + 0.05 * consistency_loss
        
        return total_loss, {
            'main_loss': main_loss,
            'aux_loss': aux_loss, 
            'consistency_loss': consistency_loss
        }
    
    def _focal_smooth_strategy(self, main_logits, aux_logits, labels):
        """
        Focal+平滑策略：针对严重不平衡的医疗数据
        """
        # 主损失：标准Focal Loss
        main_loss = self.focal_loss(main_logits, labels)
        
        # 辅助损失：更强的标签平滑 + 温度缩放
        temperature = 2.0  # 软化概率分布
        aux_loss = F.cross_entropy(aux_logits / temperature, labels, label_smoothing=0.2)
        
        # 多样性损失：鼓励两个头学习不同的表示
        main_probs = F.softmax(main_logits, dim=1)
        aux_probs = F.softmax(aux_logits, dim=1)
        diversity_loss = -F.kl_div(
            F.log_softmax(aux_logits, dim=1),
            main_probs.detach(),
            reduction='batchmean'
        )  # 负KL散度鼓励多样性
        
        # 组合损失 (60% + 30% + 10%)
        total_loss = 0.6 * main_loss + 0.3 * aux_loss + 0.1 * diversity_loss
        
        return total_loss, {
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'diversity_loss': diversity_loss
        }
    
    def _confidence_aware_strategy(self, main_logits, aux_logits, labels):
        """
        置信度感知策略：根据预测置信度动态调整损失权重
        """
        # 计算主头的置信度
        main_probs = F.softmax(main_logits, dim=1)
        main_confidence = torch.max(main_probs, dim=1)[0]  # 最大概率作为置信度
        
        # 主损失：Focal Loss
        main_loss = self.focal_loss(main_logits, labels)
        
        # 辅助损失：对低置信度样本给予更多权重
        aux_loss_raw = F.cross_entropy(aux_logits, labels, reduction='none')
        confidence_weights = 1.0 - main_confidence.detach()  # 低置信度→高权重
        aux_loss = (aux_loss_raw * confidence_weights).mean()
        
        # 校准损失：提高概率校准质量
        calibration_loss = F.mse_loss(main_confidence, (main_probs.argmax(dim=1) == labels).float())
        
        # 动态权重组合
        avg_confidence = main_confidence.mean()
        main_weight = 0.5 + 0.3 * avg_confidence  # 高置信度→更多主损失权重
        aux_weight = 0.5 - 0.2 * avg_confidence   # 低置信度→更多辅助损失权重
        
        total_loss = main_weight * main_loss + aux_weight * aux_loss + 0.1 * calibration_loss
        
        return total_loss, {
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'calibration_loss': calibration_loss,
            'avg_confidence': avg_confidence
        }
    
    def _adversarial_strategy(self, main_logits, aux_logits, labels):
        """
        对抗策略：辅助头尝试"对抗"主头，提升鲁棒性
        """
        # 主损失：Focal Loss
        main_loss = self.focal_loss(main_logits, labels)
        
        # 辅助损失：标准交叉熵
        aux_loss = F.cross_entropy(aux_logits, labels)
        
        # 对抗损失：最大化主辅助头预测的差异（在正确预测的前提下）
        main_probs = F.softmax(main_logits, dim=1)
        aux_probs = F.softmax(aux_logits, dim=1)
        
        # 只对正确预测的样本计算对抗损失
        main_correct = (main_logits.argmax(dim=1) == labels)
        aux_correct = (aux_logits.argmax(dim=1) == labels)
        both_correct = main_correct & aux_correct
        
        if both_correct.sum() > 0:
            adversarial_loss = -F.kl_div(
                F.log_softmax(aux_logits[both_correct], dim=1),
                main_probs[both_correct].detach(),
                reduction='batchmean'
            )
        else:
            adversarial_loss = torch.tensor(0.0, device=main_logits.device)
        
        # 组合损失
        total_loss = 0.6 * main_loss + 0.3 * aux_loss + 0.1 * adversarial_loss
        
        return total_loss, {
            'main_loss': main_loss,
            'aux_loss': aux_loss,
            'adversarial_loss': adversarial_loss,
            'both_correct_ratio': both_correct.float().mean()
        }


# 配置不同策略的推荐参数
STRATEGY_CONFIGS = {
    "balanced": {
        "description": "平衡策略 - 适用于轻度到中度不平衡数据",
        "focal_alpha": 0.2,
        "focal_gamma": 2.5,
        "best_for": "一般医疗分类任务"
    },
    "focal_smooth": {
        "description": "Focal+平滑策略 - 适用于严重不平衡数据", 
        "focal_alpha": 0.15,
        "focal_gamma": 3.0,
        "best_for": "罕见疾病检测，严重不平衡数据"
    },
    "confidence_aware": {
        "description": "置信度感知策略 - 适用于需要概率校准的任务",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "best_for": "风险评估，需要准确概率的任务"
    },
    "adversarial": {
        "description": "对抗策略 - 适用于需要提升鲁棒性的任务",
        "focal_alpha": 0.2,
        "focal_gamma": 2.5,
        "best_for": "噪声数据，提升模型鲁棒性"
    }
}


def get_recommended_strategy(class_imbalance_ratio, noise_level="low", requires_calibration=False):
    """
    根据数据特征推荐最佳策略
    
    Args:
        class_imbalance_ratio: 类别不平衡比例 (minority/majority)
        noise_level: 数据噪声水平 ("low", "medium", "high")
        requires_calibration: 是否需要概率校准
        
    Returns:
        strategy_name: 推荐的策略名称
    """
    if requires_calibration:
        return "confidence_aware"
    elif class_imbalance_ratio < 0.1:  # 严重不平衡
        return "focal_smooth"
    elif noise_level == "high":
        return "adversarial"
    else:
        return "balanced"  # 默认选择


# 使用示例
if __name__ == "__main__":
    print("🎯 高级损失策略配置")
    print("=" * 50)
    
    for strategy, config in STRATEGY_CONFIGS.items():
        print(f"\n📋 {strategy.upper()}:")
        print(f"   描述: {config['description']}")
        print(f"   适用于: {config['best_for']}")
        print(f"   参数: α={config['focal_alpha']}, γ={config['focal_gamma']}")
    
    print(f"\n🔧 推荐策略示例:")
    print(f"   轻度不平衡 (1:5): {get_recommended_strategy(0.2)}")
    print(f"   中度不平衡 (1:10): {get_recommended_strategy(0.1)}")
    print(f"   严重不平衡 (1:50): {get_recommended_strategy(0.02)}")
    print(f"   需要概率校准: {get_recommended_strategy(0.2, requires_calibration=True)}")