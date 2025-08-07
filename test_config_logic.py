#!/usr/bin/env python3
"""
测试自适应设备选择配置逻辑（不依赖PyTorch）
"""

import os
import sys

def mock_torch_cuda_available():
    """模拟torch.cuda.is_available()"""
    # 检查环境变量或系统信息来判断是否有GPU
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        return True
    
    # 检查是否有NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def mock_torch_cuda_device_count():
    """模拟torch.cuda.device_count()"""
    if not mock_torch_cuda_available():
        return 0
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
        return 1  # 假设至少有一个GPU
    except:
        return 1

def mock_torch_cuda_get_device_name(device_id):
    """模拟torch.cuda.get_device_name()"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            if device_id < len(gpus):
                return gpus[device_id]
        return "Unknown GPU"
    except:
        return "Unknown GPU"

def mock_torch_cuda_get_device_properties(device_id):
    """模拟torch.cuda.get_device_properties()"""
    class MockProperties:
        def __init__(self, total_memory):
            self.total_memory = total_memory
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memories = result.stdout.strip().split('\n')
            if device_id < len(memories):
                # 转换为字节 (MB to bytes)
                memory_mb = int(memories[device_id])
                memory_bytes = memory_mb * 1024 * 1024
                return MockProperties(memory_bytes)
        return MockProperties(8 * 1024 * 1024 * 1024)  # 默认8GB
    except:
        return MockProperties(8 * 1024 * 1024 * 1024)  # 默认8GB

def get_optimal_device_config():
    """
    智能检测并返回最优的设备配置（模拟版本）
    自适应选择：如果有GPU就使用GPU，否则使用CPU
    """
    # 检查CUDA是否可用
    cuda_available = mock_torch_cuda_available()
    
    if cuda_available:
        # 获取GPU信息
        gpu_count = mock_torch_cuda_device_count()
        gpu_name = mock_torch_cuda_get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = mock_torch_cuda_get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
        
        print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 根据GPU内存调整配置
        if gpu_memory >= 16:  # 16GB+ GPU
            devices = min(gpu_count, 2)  # 最多使用2个GPU
            precision = "16-mixed"
            batch_size_recommendation = 8
        elif gpu_memory >= 8:  # 8-16GB GPU
            devices = 1
            precision = "16-mixed"
            batch_size_recommendation = 4
        else:  # 小于8GB GPU
            devices = 1
            precision = "32"  # 使用32位精度避免内存不足
            batch_size_recommendation = 2
            
        accelerator = "gpu"
        device = "cuda"
        
        print(f"   - Using {devices} GPU(s)")
        print(f"   - Precision: {precision}")
        print(f"   - Recommended batch size: {batch_size_recommendation}")
        
    else:
        # CPU配置
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        print(f"💻 CPU detected: {cpu_count} cores")
        
        devices = 1
        accelerator = "cpu"
        precision = "32"  # CPU训练使用32位精度
        device = "cpu"
        batch_size_recommendation = 1
        
        print(f"   - Using CPU training")
        print(f"   - Precision: {precision}")
        print(f"   - Recommended batch size: {batch_size_recommendation}")
    
    return {
        'device': device,
        'accelerator': accelerator,
        'devices': devices,
        'precision': precision,
        'batch_size_recommendation': batch_size_recommendation,
        'cuda_available': cuda_available
    }

def test_adaptive_config():
    """测试自适应配置"""
    print("🧪 Testing Adaptive Device Configuration Logic")
    print("=" * 60)
    
    # 获取配置
    config = get_optimal_device_config()
    
    print(f"\n📊 Configuration Results:")
    print(f"  - Device: {config['device']}")
    print(f"  - Accelerator: {config['accelerator']}")
    print(f"  - Devices count: {config['devices']}")
    print(f"  - Precision: {config['precision']}")
    print(f"  - CUDA available: {config['cuda_available']}")
    print(f"  - Recommended batch size: {config['batch_size_recommendation']}")
    
    # 验证配置逻辑
    print(f"\n✅ Configuration Validation:")
    
    if config['cuda_available']:
        if config['accelerator'] == 'gpu' and config['device'] == 'cuda':
            print(f"  ✅ GPU configuration is correct")
        else:
            print(f"  ❌ GPU configuration is incorrect")
            return False
            
        if config['precision'] in ['16-mixed', '32']:
            print(f"  ✅ GPU precision setting is valid")
        else:
            print(f"  ❌ GPU precision setting is invalid")
            return False
    else:
        if config['accelerator'] == 'cpu' and config['device'] == 'cpu':
            print(f"  ✅ CPU configuration is correct")
        else:
            print(f"  ❌ CPU configuration is incorrect")
            return False
            
        if config['precision'] == '32':
            print(f"  ✅ CPU precision setting is valid")
        else:
            print(f"  ❌ CPU precision setting is invalid")
            return False
    
    print(f"\n🎉 All configuration tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Adaptive Configuration Logic Tests")
    
    # 运行配置测试
    if test_adaptive_config():
        print(f"\n🎉 All adaptive configuration logic tests completed successfully!")
    else:
        print(f"\n❌ Configuration tests failed!")
        sys.exit(1)