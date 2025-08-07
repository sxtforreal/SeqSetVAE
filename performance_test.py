#!/usr/bin/env python3
"""
性能测试脚本
用于比较不同配置的训练速度
"""

import time
import torch
import psutil
import os
import argparse
from datetime import datetime
import subprocess
import json

def get_system_info():
    """获取系统信息"""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info

def run_training_test(config_name, config_args, duration_minutes=5):
    """运行训练测试"""
    print(f"\n🧪 测试配置: {config_name}")
    print("=" * 50)
    
    # 构建命令
    cmd = ["python", "train_optimized.py"] + config_args + ["--max_epochs", "1"]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024**3)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
    
    try:
        # 运行训练
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 监控进程
        while process.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > duration_minutes * 60:
                print(f"⏰ 测试时间达到 {duration_minutes} 分钟，停止测试")
                process.terminate()
                break
            
            # 每30秒输出一次状态
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                current_memory = psutil.virtual_memory().used / (1024**3)
                memory_usage = current_memory - start_memory
                
                status = f"⏱️  已运行: {elapsed:.1f}s, 内存使用: +{memory_usage:.1f}GB"
                
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_usage = current_gpu_memory - start_gpu_memory
                    status += f", GPU内存: +{gpu_memory_usage:.1f}GB"
                
                print(status)
            
            time.sleep(1)
        
        # 获取输出
        stdout, stderr = process.communicate()
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算内存使用
        end_memory = psutil.virtual_memory().used / (1024**3)
        memory_usage = end_memory - start_memory
        
        gpu_memory_usage = 0
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_usage = end_gpu_memory - start_gpu_memory
        
        # 解析输出中的关键信息
        metrics = {
            "config_name": config_name,
            "total_time": total_time,
            "memory_usage_gb": memory_usage,
            "gpu_memory_usage_gb": gpu_memory_usage,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        print(f"✅ 测试完成 - 总时间: {total_time:.1f}s")
        print(f"   - 内存使用: +{memory_usage:.1f}GB")
        if torch.cuda.is_available():
            print(f"   - GPU内存使用: +{gpu_memory_usage:.1f}GB")
        
        return metrics
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return {
            "config_name": config_name,
            "error": str(e),
            "total_time": time.time() - start_time
        }

def main():
    parser = argparse.ArgumentParser(description="性能测试脚本")
    parser.add_argument("--duration", type=int, default=5, help="每个测试的最大运行时间（分钟）")
    parser.add_argument("--output", type=str, default="performance_results.json", help="结果输出文件")
    args = parser.parse_args()
    
    print("🚀 开始性能测试")
    print("=" * 60)
    
    # 获取系统信息
    system_info = get_system_info()
    print("📊 系统信息:")
    print(f"   - CPU核心数: {system_info['cpu_count']}")
    print(f"   - 内存: {system_info['memory_gb']:.1f}GB")
    print(f"   - GPU可用: {system_info['gpu_available']}")
    if system_info['gpu_available']:
        print(f"   - GPU数量: {system_info['gpu_count']}")
        print(f"   - GPU名称: {system_info['gpu_name']}")
        print(f"   - GPU内存: {system_info['gpu_memory_gb']:.1f}GB")
    
    # 定义测试配置
    test_configs = {
        "baseline": {
            "args": ["--batch_size", "1", "--num_workers", "2", "--precision", "32"]
        },
        "optimized_fast": {
            "args": [
                "--batch_size", "4",
                "--num_workers", "8",
                "--pin_memory",
                "--gradient_accumulation_steps", "2",
                "--max_sequence_length", "1000",
                "--precision", "16-mixed",
                "--fast_detection"
            ]
        },
        "optimized_max": {
            "args": [
                "--batch_size", "4",
                "--num_workers", "8",
                "--pin_memory",
                "--gradient_accumulation_steps", "2",
                "--max_sequence_length", "1000",
                "--precision", "16-mixed",
                "--disable_metrics_monitoring",
                "--compile_model"
            ]
        },
        "memory_efficient": {
            "args": [
                "--batch_size", "2",
                "--num_workers", "4",
                "--max_sequence_length", "500",
                "--precision", "16-mixed",
                "--fast_detection"
            ]
        }
    }
    
    # 运行测试
    results = {
        "system_info": system_info,
        "test_time": datetime.now().isoformat(),
        "duration_minutes": args.duration,
        "configs": {}
    }
    
    for config_name, config in test_configs.items():
        print(f"\n{'='*60}")
        metrics = run_training_test(config_name, config["args"], args.duration)
        results["configs"][config_name] = metrics
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("📊 测试结果总结:")
    print("=" * 60)
    
    baseline_time = results["configs"]["baseline"]["total_time"]
    
    for config_name, metrics in results["configs"].items():
        if "error" not in metrics:
            speedup = baseline_time / metrics["total_time"] if metrics["total_time"] > 0 else 0
            print(f"\n{config_name}:")
            print(f"   - 总时间: {metrics['total_time']:.1f}s")
            print(f"   - 速度提升: {speedup:.2f}x")
            print(f"   - 内存使用: +{metrics['memory_usage_gb']:.1f}GB")
            if metrics['gpu_memory_usage_gb'] > 0:
                print(f"   - GPU内存使用: +{metrics['gpu_memory_usage_gb']:.1f}GB")
    
    print(f"\n💾 详细结果已保存到: {args.output}")

if __name__ == "__main__":
    main()