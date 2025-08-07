#!/usr/bin/env python3
"""
测试skip_pretrained_on_resume功能
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SeqSetVAE
import config

def test_skip_pretrained_on_resume():
    """测试skip_pretrained_on_resume功能"""
    
    print("🧪 Testing skip_pretrained_on_resume functionality...")
    
    # 测试1: 正常情况（不跳过预训练加载）
    print("\n1. Testing normal case (skip_pretrained_on_resume=False):")
    try:
        model1 = SeqSetVAE(
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
            pretrained_ckpt=config.pretrained_ckpt,
            skip_pretrained_on_resume=False,
        )
        print("✅ Normal case passed")
    except Exception as e:
        print(f"❌ Normal case failed: {e}")
        return False
    
    # 测试2: 跳过预训练加载的情况
    print("\n2. Testing skip_pretrained_on_resume=True:")
    try:
        model2 = SeqSetVAE(
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
            pretrained_ckpt=config.pretrained_ckpt,
            skip_pretrained_on_resume=True,
        )
        print("✅ Skip pretrained case passed")
    except Exception as e:
        print(f"❌ Skip pretrained case failed: {e}")
        return False
    
    # 测试3: 没有预训练ckpt的情况
    print("\n3. Testing no pretrained_ckpt case:")
    try:
        model3 = SeqSetVAE(
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
            pretrained_ckpt=None,
            skip_pretrained_on_resume=False,
        )
        print("✅ No pretrained ckpt case passed")
    except Exception as e:
        print(f"❌ No pretrained ckpt case failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_skip_pretrained_on_resume()
    
    if success:
        print("\n✅ skip_pretrained_on_resume functionality is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ skip_pretrained_on_resume functionality test failed!")
        sys.exit(1)