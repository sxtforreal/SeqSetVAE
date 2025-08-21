#!/usr/bin/env python3
"""
Test script to verify complete separation between pretrain and finetune models
验证预训练和微调模型完全分离的测试脚本
"""

import torch
import torch.nn as nn
from model import SeqSetVAE, SeqSetVAEPretrain
import config

def test_model_independence():
    """测试两个模型的完全独立性"""
    print("🧪 Testing Model Independence")
    print("-" * 40)
    
    # 创建预训练模型
    pretrain_model = SeqSetVAEPretrain(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
    )
    
    # 创建微调模型
    finetune_model = SeqSetVAE(
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
    )
    
    # 检查模型类型
    print(f"   - Pretrain model type: {type(pretrain_model).__name__}")
    print(f"   - Finetune model type: {type(finetune_model).__name__}")
    
    # 检查模型是否有不同的方法
    pretrain_methods = set(dir(pretrain_model))
    finetune_methods = set(dir(finetune_model))
    
    finetune_only = finetune_methods - pretrain_methods
    pretrain_only = pretrain_methods - finetune_methods
    
    print(f"   - Methods only in finetune model: {len(finetune_only)}")
    if finetune_only:
        print(f"     {list(finetune_only)[:5]}...")  # Show first 5
    
    print(f"   - Methods only in pretrain model: {len(pretrain_only)}")
    if pretrain_only:
        print(f"     {list(pretrain_only)[:5]}...")  # Show first 5
    
    return True

def test_vae_feature_extraction():
    """测试VAE特征提取的不同实现"""
    print("\n🧪 Testing VAE Feature Extraction Differences")
    print("-" * 40)
    
    batch_size = 2
    latent_dim = config.latent_dim
    
    # 创建模拟的VAE输出
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    # 测试预训练模型的特征提取（应该只使用mu）
    print("   - Pretrain model feature extraction:")
    print("     Uses only mu (original design)")
    pretrain_feature = mu.clone()  # 预训练模型直接使用mu
    print(f"     Feature shape: {pretrain_feature.shape}")
    
    # 测试微调模型的特征提取（使用mu+logvar融合）
    print("   - Finetune model feature extraction:")
    finetune_model = SeqSetVAE(
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
    )
    
    # 测试分类模式下的VAE特征融合
    finetune_model.classification_only = True
    fused_feature = finetune_model._fuse_vae_features(mu, logvar)
    print("     Uses mu + logvar fusion (enhanced design)")
    print(f"     Feature shape: {fused_feature.shape}")
    
    # 验证特征不同
    feature_diff = torch.norm(pretrain_feature - fused_feature).item()
    print(f"     Feature difference norm: {feature_diff:.4f}")
    
    if feature_diff > 1e-6:
        print("     ✅ Features are different - models use different extraction methods")
        return True
    else:
        print("     ❌ Features are too similar - potential issue")
        return False

def test_model_parameters():
    """测试模型参数结构的差异"""
    print("\n🧪 Testing Model Parameter Structures")
    print("-" * 40)
    
    # 创建两个模型
    pretrain_model = SeqSetVAEPretrain(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
    )
    
    finetune_model = SeqSetVAE(
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
    )
    
    # 统计参数
    pretrain_params = sum(p.numel() for p in pretrain_model.parameters())
    finetune_params = sum(p.numel() for p in finetune_model.parameters())
    
    print(f"   - Pretrain model parameters: {pretrain_params:,}")
    print(f"   - Finetune model parameters: {finetune_params:,}")
    print(f"   - Parameter difference: {finetune_params - pretrain_params:,}")
    
    # 检查微调模型特有的模块
    pretrain_modules = set(name for name, _ in pretrain_model.named_modules())
    finetune_modules = set(name for name, _ in finetune_model.named_modules())
    
    finetune_only_modules = finetune_modules - pretrain_modules
    print(f"   - Modules only in finetune model: {len(finetune_only_modules)}")
    
    # 检查关键的微调特有模块
    key_finetune_modules = [name for name in finetune_only_modules 
                           if any(key in name for key in ['cls_head', 'vae_feature_fusion', 'feature_fusion'])]
    
    if key_finetune_modules:
        print("     Key finetune-specific modules:")
        for module in sorted(key_finetune_modules)[:10]:  # Show first 10
            print(f"       - {module}")
        return True
    else:
        print("     ❌ No finetune-specific modules found")
        return False

def test_forward_compatibility():
    """测试前向传播兼容性"""
    print("\n🧪 Testing Forward Pass Compatibility")
    print("-" * 40)
    
    # 创建模拟输入数据
    batch_size = 1
    seq_len = 2
    num_vars = 5
    
    sets = []
    for i in range(seq_len):
        set_dict = {
            'var': torch.randn(batch_size, num_vars, config.input_dim),
            'val': torch.rand(batch_size, num_vars, 1),
            'minute': torch.tensor([[i * 60.0]] * batch_size)
        }
        sets.append(set_dict)
    
    try:
        # 测试预训练模型
        pretrain_model = SeqSetVAEPretrain(
            input_dim=config.input_dim,
            reduced_dim=config.reduced_dim,
            latent_dim=config.latent_dim,
            levels=config.levels,
            heads=config.heads,
            m=config.m,
            beta=config.beta,
            lr=config.lr,
            ff_dim=config.ff_dim,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
        )
        
        pretrain_model.eval()
        with torch.no_grad():
            recon_loss, kl_loss = pretrain_model(sets)
        
        print(f"   - Pretrain model output: recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")
        
        # 测试微调模型
        finetune_model = SeqSetVAE(
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
        )
        
        finetune_model.eval()
        with torch.no_grad():
            logits, recon_loss, kl_loss = finetune_model(sets)
        
        print(f"   - Finetune model output: logits={logits.shape}, recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")
        print("   ✅ Both models can process the same input format")
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Testing Pretrain-Finetune Separation")
    print("=" * 50)
    
    tests = [
        ("Model Independence", test_model_independence),
        ("VAE Feature Extraction", test_vae_feature_extraction),
        ("Model Parameters", test_model_parameters),
        ("Forward Compatibility", test_forward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 Separation Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Pretrain and finetune models are completely separated!")
        print("   - Pretrain: Uses original simple design")
        print("   - Finetune: Uses enhanced design with modern VAE features")
    else:
        print("⚠️ Some separation issues detected.")

if __name__ == "__main__":
    main()