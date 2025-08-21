#!/usr/bin/env python3
"""
SeqSetVAE Finetune Debugging and Diagnosis Script
用于诊断分类性能问题的调试脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import argparse
import os
from model import SeqSetVAE, load_checkpoint_weights
from dataset import SeqSetVAEDataModule
import config

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

def analyze_feature_quality(model, dataloader, device, max_batches=10):
    """分析特征质量"""
    print("\n🔍 Feature Quality Analysis:")
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            try:
                # Forward pass to get features
                logits, _, _ = model(batch)
                
                # Get features before final classification layer
                # This requires modifying the model to expose intermediate features
                # For now, we'll use the logits as a proxy
                all_features.append(logits.cpu().numpy())
                all_labels.append(batch['label'].cpu().numpy())
                
            except Exception as e:
                print(f"   ⚠️ Error processing batch {i}: {e}")
                continue
    
    if all_features:
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"   - Feature shape: {features.shape}")
        print(f"   - Feature mean: {features.mean():.4f}")
        print(f"   - Feature std: {features.std():.4f}")
        print(f"   - Label distribution: {np.bincount(labels)}")
        
        # Calculate class separation
        if len(np.unique(labels)) == 2:
            class_0_features = features[labels == 0]
            class_1_features = features[labels == 1]
            
            if len(class_0_features) > 0 and len(class_1_features) > 0:
                separation = np.abs(class_0_features.mean() - class_1_features.mean())
                print(f"   - Class separation: {separation:.4f}")

def analyze_gradients(model):
    """分析梯度情况"""
    print("\n🔍 Gradient Analysis:")
    
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if name.startswith('cls_head'):
                print(f"   - {name}: grad_norm={param_norm.item():.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"   - Total gradient norm: {total_norm:.6f}")
    print(f"   - Parameters with gradients: {param_count}")

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
    parser = argparse.ArgumentParser(description="Debug SeqSetVAE finetune performance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--params_map", type=str, required=True, help="Path to params map")
    parser.add_argument("--label_file", type=str, required=True, help="Path to label file")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    
    print("🔧 SeqSetVAE Finetune Debugging Tool")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\n📦 Loading model from: {args.checkpoint}")
    
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
        w=config.w,
        free_bits=config.free_bits,
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    model.to(device)
    
    # Setup data
    print(f"\n📊 Setting up data from: {args.data_dir}")
    
    data_module = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map,
        label_path=args.label_file,
        batch_size=args.batch_size,
    )
    
    try:
        data_module.setup('fit')
        val_dataloader = data_module.val_dataloader()
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Run analysis
    analyze_model_weights(model)
    analyze_feature_quality(model, val_dataloader, device)
    evaluate_model_performance(model, val_dataloader, device)
    
    # Test a single forward pass to check for gradient flow
    print("\n🔍 Testing gradient flow:")
    model.train()
    model.enable_classification_only_mode()
    
    try:
        batch = next(iter(val_dataloader))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        logits, recon_loss, kl_loss = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch['label'])
        loss.backward()
        
        analyze_gradients(model)
        
    except Exception as e:
        print(f"   ❌ Error in gradient test: {e}")
    
    print("\n✅ Debugging analysis complete!")

if __name__ == "__main__":
    main()