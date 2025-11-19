#!/usr/bin/env python3
"""
快速测试 LLM 配置
用法: python scripts/test_llm.py
"""

import os
import sys

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("LLM 配置快速测试")
print("=" * 60)
print()

# 检查环境变量
doubao_key = os.getenv("DOUBAO_API_KEY")
llm_provider = os.getenv("LLM_PROVIDER", "doubao")

print(f"当前配置:")
print(f"  LLM_PROVIDER: {llm_provider}")
print(f"  DOUBAO_API_KEY: {'已设置' if doubao_key else '❌ 未设置'}")
print()

if not doubao_key:
    print("❌ 错误: DOUBAO_API_KEY 未设置")
    print()
    print("请使用以下方法之一设置:")
    print()
    print("方法一：在当前终端设置（推荐用于测试）")
    print("  export DOUBAO_API_KEY=your_actual_api_key")
    print("  export LLM_PROVIDER=doubao")
    print("  python scripts/test_llm.py")
    print()
    print("方法二：创建 .env 文件（推荐用于项目）")
    print("  echo 'DOUBAO_API_KEY=your_actual_api_key' > .env")
    print("  echo 'LLM_PROVIDER=doubao' >> .env")
    print("  python scripts/test_llm.py")
    print()
    sys.exit(1)

# 测试 LLM
try:
    from llm.llm_client import get_llm_client
    
    print("正在初始化 LLM 客户端...")
    client = get_llm_client(provider=llm_provider)
    print(f"✅ 客户端初始化成功")
    print(f"   提供商: {llm_provider}")
    print(f"   模型: {client.model_name}")
    print()
    
    print("正在测试 API 调用...")
    response = client.generate("你好，请用一句话介绍你自己", max_tokens=50)
    print(f"✅ API 调用成功!")
    print(f"   响应: {response}")
    print()
    print("=" * 60)
    print("✅ 配置验证成功！可以正常使用 LLM")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ 错误: {e}")
    print()
    print("可能的原因:")
    print("  1. API Key 格式不正确")
    print("  2. API Key 无效或已过期")
    print("  3. 网络连接问题")
    print("  4. 豆包服务未开通")
    print()
    print("请检查:")
    print("  1. API Key 是否正确（不要包含引号）")
    print("  2. 是否在火山引擎开通了豆包服务")
    print("  3. 网络是否可以访问火山引擎 API")
    sys.exit(1)

