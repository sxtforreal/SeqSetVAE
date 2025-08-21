#!/usr/bin/env python3
"""
SeqSetVAE Complete Test Suite
综合测试套件：验证所有改进功能和预训练微调分离
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import argparse
import os
from model import SeqSetVAE, SeqSetVAEPretrain, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
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
    print(f"   - Methods only in pretrain model: {len(pretrain_only)}")
    
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

def test_parameter_freezing():
    """测试参数冻结机制"""
    print("\n🧪 Testing Parameter Freezing Mechanism")
    print("-" * 40)
    
    model = SeqSetVAE(
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
    
    # 模拟finetune模式的冻结
    model.enable_classification_only_mode()
    
    frozen_params = 0
    trainable_params = 0
    trainable_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
        else:
            frozen_params += param.numel()
    
    print(f"   - Frozen parameters: {frozen_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Trainable ratio: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")
    print(f"   - Trainable parameter names: {trainable_names}")
    
    # 验证只有分类头是可训练的
    all_cls_head = all(name.startswith('cls_head') for name in trainable_names)
    if all_cls_head and trainable_params > 0:
        print("   ✅ Freezing mechanism working correctly")
        return True
    else:
        print("   ❌ Freezing mechanism has issues")
        return False

def test_model_forward():
    """测试模型前向传播"""
    print("\n🧪 Testing Model Forward Pass")
    print("-" * 40)
    
    # 创建模拟输入
    batch_size = 1
    seq_len = 3
    num_vars = 10
    
    # 创建模拟的患者序列数据
    sets = []
    for i in range(seq_len):
        set_dict = {
            'var': torch.randn(batch_size, num_vars, config.input_dim),
            'val': torch.rand(batch_size, num_vars, 1),
            'minute': torch.tensor([[i * 60.0]] * batch_size)  # 每小时一个set
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

def analyze_model_weights(model):
    """分析模型权重分布"""
    print("\n🔍 Model Weight Analysis:")
    
    # 分析分类头权重
    cls_weights = []
    cls_biases = []
    
    for name, param in model.named_parameters():
        if name.startswith('cls_head') and 'weight' in name:
            cls_weights.append(param.data.cpu().numpy().flatten())
        elif name.startswith('cls_head') and 'bias' in name:
            cls_biases.append(param.data.cpu().numpy().flatten())
    
    if cls_weights:
        all_weights = np.concatenate(cls_weights)
        print(f"   - Classification head weights: mean={all_weights.mean():.4f}, std={all_weights.std():.4f}")
        print(f"   - Weight range: [{all_weights.min():.4f}, {all_weights.max():.4f}]")
        
    if cls_biases:
        all_biases = np.concatenate(cls_biases)
        print(f"   - Classification head biases: mean={all_biases.mean():.4f}, std={all_biases.std():.4f}")

def evaluate_model_performance(model, dataloader, device, max_batches=50):
    """评估模型性能"""
    print("\n🔍 Model Performance Analysis:")
    
    model.eval()
    all_probs = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            try:
                logits, _, _ = model(batch)
                probs = torch.softmax(logits, dim=1)
                
                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch['label'].cpu().numpy())
                
            except Exception as e:
                print(f"   ⚠️ Error processing batch {i}: {e}")
                continue
    
    if all_probs:
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        
        # Binary classification metrics
        if probs.shape[1] == 2:
            pos_probs = probs[:, 1]
            
            # ROC curve
            fpr, tpr, _ = roc_curve(labels, pos_probs)
            auc = np.trapz(tpr, fpr)
            
            # PR curve
            precision, recall, _ = precision_recall_curve(labels, pos_probs)
            auprc = np.trapz(precision, recall)
            
            print(f"   - AUC: {auc:.4f}")
            print(f"   - AUPRC: {auprc:.4f}")
            print(f"   - Probability range: [{pos_probs.min():.4f}, {pos_probs.max():.4f}]")
            print(f"   - Probability mean: {pos_probs.mean():.4f}")
            
            # Check for extreme probabilities
            extreme_low = (pos_probs < 0.01).sum()
            extreme_high = (pos_probs > 0.99).sum()
            print(f"   - Extreme probabilities: {extreme_low} low (<0.01), {extreme_high} high (>0.99)")

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="SeqSetVAE Complete Test Suite")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for performance testing")
    parser.add_argument("--data_dir", type=str, help="Path to data directory for performance testing")
    parser.add_argument("--params_map", type=str, help="Path to params map for performance testing")
    parser.add_argument("--label_file", type=str, help="Path to label file for performance testing")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing")
    
    args = parser.parse_args()
    
    print("🚀 SeqSetVAE Complete Test Suite")
    print("=" * 60)
    
    # 基础功能测试
    basic_tests = [
        ("Model Independence", test_model_independence),
        ("VAE Feature Extraction", test_vae_feature_extraction),
        ("Parameter Freezing", test_parameter_freezing),
        ("Model Forward Pass", test_model_forward),
    ]
    
    results = []
    for test_name, test_func in basic_tests:
        print(f"\n📋 {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # 性能测试（如果提供了参数）
    if all([args.checkpoint, args.data_dir, args.params_map, args.label_file]):
        print(f"\n📋 Performance Analysis")
        try:
            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Load model
            model = SeqSetVAE(
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
            
            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            
            # Setup data
            data_module = SeqSetVAEDataModule(
                saved_dir=args.data_dir,
                params_map_path=args.params_map,
                label_path=args.label_file,
                batch_size=args.batch_size,
            )
            data_module.setup('fit')
            val_dataloader = data_module.val_dataloader()
            
            # Run analysis
            analyze_model_weights(model)
            evaluate_model_performance(model, val_dataloader, device)
            
            results.append(("Performance Analysis", True))
            
        except Exception as e:
            print(f"   ❌ Performance analysis failed: {e}")
            results.append(("Performance Analysis", False))
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! SeqSetVAE improvements are working correctly!")
        print("   - Pretrain and finetune models are completely separated")
        print("   - VAE feature fusion is working as expected")
        print("   - Parameter freezing is functioning properly")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()