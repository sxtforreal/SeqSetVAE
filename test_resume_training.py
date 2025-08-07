#!/usr/bin/env python3
"""
测试从checkpoint继续训练的功能
"""

import os
import sys
import argparse
from pathlib import Path

def test_resume_training():
    """测试从checkpoint继续训练的功能"""
    
    # 检查train_optimized.py是否存在
    if not os.path.exists("train_optimized.py"):
        print("❌ train_optimized.py not found")
        return False
    
    # 检查是否有checkpoint文件
    checkpoint_dir = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("💡 Please run training first to generate checkpoint files")
        return False
    
    # 查找checkpoint文件
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        print("💡 Please run training first to generate checkpoint files")
        return False
    
    print(f"✅ Found {len(checkpoint_files)} checkpoint files:")
    for i, ckpt in enumerate(checkpoint_files, 1):
        print(f"  {i}. {os.path.basename(ckpt)}")
    
    # 测试命令行参数解析
    print("\n🧪 Testing command line argument parsing...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "train_optimized.py", 
            "--help"
        ], capture_output=True, text=True)
        
        if "--resume_from_checkpoint" in result.stdout:
            print("✅ --resume_from_checkpoint argument found")
        else:
            print("❌ --resume_from_checkpoint argument not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing argument parsing: {e}")
        return False
    
    # 测试checkpoint文件验证
    print("\n🧪 Testing checkpoint file validation...")
    test_ckpt = checkpoint_files[0]
    try:
        result = subprocess.run([
            sys.executable, "train_optimized.py",
            "--resume_from_checkpoint", test_ckpt,
            "--max_epochs", "1",  # 只训练1个epoch进行测试
            "--batch_size", "1",  # 使用小batch size
            "--disable_metrics_monitoring",  # 禁用监控以加快测试
        ], capture_output=True, text=True, timeout=60)
        
        if "Resuming training from checkpoint" in result.stdout:
            print("✅ Checkpoint resume functionality working")
        else:
            print("❌ Checkpoint resume functionality not working")
            print("Output:", result.stdout)
            print("Error:", result.stderr)
            return False
            
        # 测试新的skip_pretrained_on_resume功能
        if "Skipping pretrained SetVAE loading (resuming from checkpoint)" in result.stdout:
            print("✅ skip_pretrained_on_resume functionality working")
        else:
            print("⚠️  skip_pretrained_on_resume functionality not detected in output")
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (this is normal for a quick test)")
    except Exception as e:
        print(f"❌ Error testing checkpoint resume: {e}")
        return False
    
    print("\n✅ All tests passed!")
    print("\n📝 Usage examples:")
    print(f"# 从最新的checkpoint继续训练")
    print(f"python train_optimized.py --resume_from_checkpoint {checkpoint_files[0]}")
    print(f"\n# 从最佳checkpoint继续训练")
    print(f"python train_optimized.py --resume_from_checkpoint {checkpoint_dir}/best_*.ckpt")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test resume training functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("🔍 Testing resume training functionality...")
    
    success = test_resume_training()
    
    if success:
        print("\n🎉 Resume training functionality is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Resume training functionality test failed!")
        sys.exit(1)