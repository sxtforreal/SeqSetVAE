#!/usr/bin/env python3
"""
Test script for batch training functionality.
This script tests the new batch training capabilities with dynamic padding.
"""

import torch
import lightning.pytorch as pl
from dataset import SeqSetVAEDataModule
from model import SeqSetVAE
import config
import os

def test_batch_training():
    """Test batch training functionality"""
    
    print("🧪 Testing batch training functionality...")
    print("=" * 50)
    
    # Test parameters
    test_batch_sizes = [1, 2, 4]
    data_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    
    for batch_size in test_batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")
        print("-" * 30)
        
        try:
            # Create data module
            data_module = SeqSetVAEDataModule(
                saved_dir=data_dir,
                params_map_path=params_map_path,
                label_path=label_path,
                batch_size=batch_size,
                max_sequence_length=1000,  # Limit sequence length for testing
                use_dynamic_padding=True
            )
            data_module.setup()
            
            # Create model
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
                freeze_ratio=0.0,
                pretrained_ckpt=config.pretrained_ckpt,
                w=config.w,
                free_bits=config.free_bits,
                warmup_beta=config.warmup_beta,
                max_beta=config.max_beta,
                beta_warmup_steps=config.beta_warmup_steps,
                kl_annealing=config.kl_annealing,
            )
            
            # Test data loading
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            print(f"  - Training batches: {len(train_loader)}")
            print(f"  - Validation batches: {len(val_loader)}")
            
            # Test a few batches
            for i, batch in enumerate(train_loader):
                if i >= 2:  # Only test first 2 batches
                    break
                    
                print(f"  - Batch {i+1}:")
                print(f"    - var shape: {batch['var'].shape}")
                print(f"    - val shape: {batch['val'].shape}")
                print(f"    - minute shape: {batch['minute'].shape}")
                print(f"    - set_id shape: {batch['set_id'].shape}")
                print(f"    - label shape: {batch['label'].shape}")
                
                if 'padding_mask' in batch:
                    print(f"    - padding_mask shape: {batch['padding_mask'].shape}")
                    print(f"    - padding_mask sum: {batch['padding_mask'].sum().item()}")
                
                # Test forward pass
                try:
                    with torch.no_grad():
                        logits, recon_loss, kl_loss = model(batch)
                        print(f"    - logits shape: {logits.shape}")
                        print(f"    - recon_loss: {recon_loss.item():.4f}")
                        print(f"    - kl_loss: {kl_loss.item():.4f}")
                        print(f"    ✅ Forward pass successful")
                except Exception as e:
                    print(f"    ❌ Forward pass failed: {e}")
                    break
            
            print(f"  ✅ Batch size {batch_size} test completed successfully")
            
        except Exception as e:
            print(f"  ❌ Batch size {batch_size} test failed: {e}")
            continue
    
    print("\n🎉 Batch training test completed!")

def test_memory_usage():
    """Test memory usage with different batch sizes"""
    
    print("\n💾 Testing memory usage...")
    print("=" * 30)
    
    # Test parameters
    batch_sizes = [1, 2, 4, 8]
    data_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing memory usage with batch size: {batch_size}")
        
        try:
            # Create data module
            data_module = SeqSetVAEDataModule(
                saved_dir=data_dir,
                params_map_path=params_map_path,
                label_path=label_path,
                batch_size=batch_size,
                max_sequence_length=500,  # Limit for memory testing
                use_dynamic_padding=True
            )
            data_module.setup()
            
            # Create model
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
                freeze_ratio=0.0,
                pretrained_ckpt=config.pretrained_ckpt,
                w=config.w,
                free_bits=config.free_bits,
                warmup_beta=config.warmup_beta,
                max_beta=config.max_beta,
                beta_warmup_steps=config.beta_warmup_steps,
                kl_annealing=config.kl_annealing,
            )
            
            # Test memory usage
            train_loader = data_module.train_dataloader()
            
            # Get first batch
            batch = next(iter(train_loader))
            
            # Measure memory before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            # Forward pass
            with torch.no_grad():
                logits, recon_loss, kl_loss = model(batch)
            
            # Measure memory after forward pass
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                
                print(f"  - Memory allocated: {(end_memory - start_memory) / 1024**2:.2f} MB")
                print(f"  - Peak memory: {peak_memory / 1024**2:.2f} MB")
                print(f"  - Batch memory efficiency: {peak_memory / (batch_size * 1024**2):.2f} MB per sample")
            
            print(f"  ✅ Memory test completed for batch size {batch_size}")
            
        except Exception as e:
            print(f"  ❌ Memory test failed for batch size {batch_size}: {e}")
            continue

if __name__ == "__main__":
    # Test basic functionality
    test_batch_training()
    
    # Test memory usage
    test_memory_usage()
    
    print("\n🎯 All tests completed!")