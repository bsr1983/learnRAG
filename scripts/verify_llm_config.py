#!/usr/bin/env python3
"""
验证 LLM 配置脚本
检查环境变量和 LLM 客户端是否正常工作
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

print("=" * 60)
print("LLM 配置验证")
print("=" * 60)
print()

# 1. 检查环境变量
print("1. 检查环境变量:")
print("-" * 60)

providers = {
    "doubao": "DOUBAO_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "ernie": "ERNIE_API_KEY",
    "zhipu": "ZHIPU_API_KEY"
}

llm_provider = os.getenv("LLM_PROVIDER", "not set")
print(f"LLM_PROVIDER: {llm_provider}")

for provider, env_var in providers.items():
    value = os.getenv(env_var)
    if value:
        # 只显示前10个字符和后5个字符，保护隐私
        masked = value[:10] + "..." + value[-5:] if len(value) > 15 else "***"
        print(f"  ✅ {env_var}: {masked}")
    else:
        print(f"  ❌ {env_var}: 未设置")

print()

# 2. 尝试初始化 LLM 客户端
print("2. 测试 LLM 客户端初始化:")
print("-" * 60)

try:
    from llm.llm_client import get_llm_client
    
    # 如果设置了 LLM_PROVIDER，使用它；否则尝试 doubao
    provider = llm_provider if llm_provider != "not set" else "doubao"
    
    print(f"尝试使用提供商: {provider}")
    client = get_llm_client(provider=provider)
    print(f"✅ LLM 客户端初始化成功")
    print(f"   提供商: {provider}")
    print(f"   模型: {client.model_name}")
    print(f"   Base URL: {client.base_url}")
    
except Exception as e:
    print(f"❌ LLM 客户端初始化失败: {e}")
    print()
    print("可能的原因:")
    print("  1. API Key 未设置或格式不正确")
    print("  2. 提供商名称错误")
    print("  3. 网络连接问题")
    sys.exit(1)

print()

# 3. 测试 LLM 调用
print("3. 测试 LLM API 调用:")
print("-" * 60)

try:
    test_prompt = "请用一句话介绍你自己"
    print(f"测试提示: {test_prompt}")
    print("正在调用 LLM...")
    
    response = client.generate(
        prompt=test_prompt,
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"✅ LLM 调用成功!")
    print(f"响应: {response}")
    
except Exception as e:
    print(f"❌ LLM 调用失败: {e}")
    print()
    print("可能的原因:")
    print("  1. API Key 无效或已过期")
    print("  2. API 端点不可访问")
    print("  3. 模型名称不正确")
    print("  4. 网络连接问题")
    sys.exit(1)

print()
print("=" * 60)
print("✅ 所有检查通过！LLM 配置正确")
print("=" * 60)
print()
print("💡 提示:")
print("  如果环境变量未设置，请使用以下方法之一:")
print()
print("  方法一：在当前终端设置（临时）")
print("    export DOUBAO_API_KEY=your_api_key")
print("    export LLM_PROVIDER=doubao")
print()
print("  方法二：使用 .env 文件（推荐）")
print("    1. 创建 .env 文件：cp .env.example .env")
print("    2. 编辑 .env，设置 DOUBAO_API_KEY 和 LLM_PROVIDER")
print()
print("  方法三：添加到 ~/.zshrc（永久）")
print("    echo 'export DOUBAO_API_KEY=your_api_key' >> ~/.zshrc")
print("    echo 'export LLM_PROVIDER=doubao' >> ~/.zshrc")
print("    source ~/.zshrc")

