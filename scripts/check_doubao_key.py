#!/usr/bin/env python3
"""
检查 DOUBAO_API_KEY 环境变量的 Python 脚本
使用方法: python scripts/check_doubao_key.py
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=env_path)
    print("✅ 已加载 .env 文件")
except ImportError:
    print("⚠️  python-dotenv 未安装，仅检查系统环境变量")
except Exception as e:
    print(f"⚠️  加载 .env 文件时出错: {e}")

print("\n" + "=" * 60)
print("检查 DOUBAO_API_KEY 环境变量")
print("=" * 60)

# 使用 os.getenv() 读取环境变量（与代码中一致）
doubao_key = os.getenv("DOUBAO_API_KEY")

if doubao_key:
    # 只显示部分内容（保护隐私）
    if len(doubao_key) > 15:
        masked = f"{doubao_key[:10]}...{doubao_key[-5:]}"
    else:
        masked = "***" + doubao_key[-5:]
    
    print(f"\n✅ DOUBAO_API_KEY: 已设置")
    print(f"   部分显示: {masked}")
    print(f"   完整值: {doubao_key}")
    print(f"   长度: {len(doubao_key)} 字符")
else:
    print(f"\n❌ DOUBAO_API_KEY: 未设置")
    print("\n设置方法:")
    print("1. 命令行临时设置:")
    print("   export DOUBAO_API_KEY=your_key")
    print("\n2. 在 .env 文件中设置:")
    print("   echo 'DOUBAO_API_KEY=your_key' >> .env")
    print("\n3. 永久设置（添加到 ~/.zshrc）:")
    print("   echo 'export DOUBAO_API_KEY=your_key' >> ~/.zshrc")
    print("   source ~/.zshrc")

print("\n" + "=" * 60)

