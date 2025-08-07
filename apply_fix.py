#!/usr/bin/env python3
"""
Apply config module import fix
"""

import os
import shutil
from datetime import datetime

def apply_config_fix():
    """应用config模块导入修复"""
    
    print("🔧 Applying config module import fix...")
    
    # 1. 备份原始文件
    backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    if os.path.exists("config.py"):
        shutil.copy2("config.py", backup_name)
        print(f"✅ Backed up original config.py to {backup_name}")
    else:
        print("❌ config.py not found")
        return False
    
    # 2. 检查修复版文件是否存在
    if not os.path.exists("config_fixed.py"):
        print("❌ config_fixed.py not found")
        return False
    
    # 3. 应用修复
    shutil.copy2("config_fixed.py", "config.py")
    print("✅ Applied config_fixed.py to config.py")
    
    # 4. 验证修复
    try:
        import config
        print("✅ Config module imported successfully")
        
        if hasattr(config, 'device_config'):
            print("✅ device_config attribute exists")
            device_config = config.device_config
            print(f"   Device: {device_config.get('device', 'N/A')}")
            print(f"   Accelerator: {device_config.get('accelerator', 'N/A')}")
        else:
            print("❌ device_config attribute still missing")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n🎉 Config module import fix applied successfully!")
    print(f"📁 Original file backed up as: {backup_name}")
    print("\n📋 Next steps:")
    print("1. Test your training scripts")
    print("2. If issues persist, restore from backup:")
    print(f"   cp {backup_name} config.py")
    
    return True

def restore_backup(backup_file):
    """从备份恢复"""
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, "config.py")
        print(f"✅ Restored config.py from {backup_file}")
    else:
        print(f"❌ Backup file {backup_file} not found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) > 2:
            restore_backup(sys.argv[2])
        else:
            print("Usage: python3 apply_fix.py restore <backup_file>")
    else:
        apply_config_fix()