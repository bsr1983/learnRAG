#!/bin/bash
# 检查 DOUBAO_API_KEY 环境变量的脚本

echo "============================================================"
echo "检查 DOUBAO_API_KEY 环境变量"
echo "============================================================"
echo

# 方法 1: 检查当前 shell 环境变量
echo "1. 当前 shell 环境变量:"
if [ -z "$DOUBAO_API_KEY" ]; then
    echo "   ❌ DOUBAO_API_KEY: 未设置"
else
    # 只显示前10个字符和后5个字符，中间用 ... 代替（保护隐私）
    masked="${DOUBAO_API_KEY:0:10}...${DOUBAO_API_KEY: -5}"
    echo "   ✅ DOUBAO_API_KEY: $masked"
    echo "   完整值: $DOUBAO_API_KEY"
fi
echo

# 方法 2: 检查 .env 文件
echo "2. .env 文件中的配置:"
if [ -f .env ]; then
    if grep -q "DOUBAO_API_KEY" .env; then
        echo "   ✅ .env 文件中包含 DOUBAO_API_KEY"
        # 显示值（隐藏敏感信息）
        key_line=$(grep "DOUBAO_API_KEY" .env | head -1)
        if [[ $key_line == *"="* ]]; then
            key_value=$(echo "$key_line" | cut -d'=' -f2- | tr -d ' ' | tr -d '"' | tr -d "'")
            if [ -n "$key_value" ]; then
                masked="${key_value:0:10}...${key_value: -5}"
                echo "   值: $masked"
            else
                echo "   ⚠️  值为空"
            fi
        fi
    else
        echo "   ❌ .env 文件中未找到 DOUBAO_API_KEY"
    fi
else
    echo "   ❌ .env 文件不存在"
fi
echo

# 方法 3: 使用 Python 检查（模拟 os.getenv）
echo "3. Python os.getenv() 读取结果:"
python3 << EOF
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

key = os.getenv("DOUBAO_API_KEY")
if key:
    masked = f"{key[:10]}...{key[-5:]}" if len(key) > 15 else "***"
    print(f"   ✅ DOUBAO_API_KEY: {masked}")
    print(f"   完整值: {key}")
else:
    print("   ❌ DOUBAO_API_KEY: 未设置")
EOF

echo
echo "============================================================"
echo "使用说明:"
echo "============================================================"
echo "1. 在命令行中查看: echo \$DOUBAO_API_KEY"
echo "2. 在命令行中设置: export DOUBAO_API_KEY=your_key"
echo "3. 在 .env 文件中设置: echo 'DOUBAO_API_KEY=your_key' >> .env"
echo "4. 永久设置（添加到 ~/.zshrc 或 ~/.bashrc）:"
echo "   echo 'export DOUBAO_API_KEY=your_key' >> ~/.zshrc"
echo "   source ~/.zshrc"
echo

