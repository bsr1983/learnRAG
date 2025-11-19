#!/bin/bash
# 检查环境变量配置脚本

echo "=========================================="
echo "环境变量检查"
echo "=========================================="
echo ""

echo "1. 检查当前 shell 的环境变量:"
echo "----------------------------------------"
if [ -z "$DOUBAO_API_KEY" ]; then
    echo "  ❌ DOUBAO_API_KEY: 未设置"
else
    masked="${DOUBAO_API_KEY:0:10}...${DOUBAO_API_KEY: -5}"
    echo "  ✅ DOUBAO_API_KEY: $masked"
fi

if [ -z "$LLM_PROVIDER" ]; then
    echo "  ❌ LLM_PROVIDER: 未设置"
else
    echo "  ✅ LLM_PROVIDER: $LLM_PROVIDER"
fi

echo ""
echo "2. 检查 .env 文件:"
echo "----------------------------------------"
if [ -f ".env" ]; then
    echo "  ✅ .env 文件存在"
    if grep -q "DOUBAO_API_KEY" .env 2>/dev/null; then
        echo "  ✅ .env 中包含 DOUBAO_API_KEY"
    else
        echo "  ❌ .env 中未找到 DOUBAO_API_KEY"
    fi
    if grep -q "LLM_PROVIDER" .env 2>/dev/null; then
        echo "  ✅ .env 中包含 LLM_PROVIDER"
    else
        echo "  ❌ .env 中未找到 LLM_PROVIDER"
    fi
else
    echo "  ❌ .env 文件不存在"
fi

echo ""
echo "3. 设置环境变量的方法:"
echo "----------------------------------------"
echo ""
echo "方法一：在当前终端设置（临时）"
echo "  export DOUBAO_API_KEY=your_api_key"
echo "  export LLM_PROVIDER=doubao"
echo ""
echo "方法二：使用 .env 文件（推荐，永久）"
echo "  1. 创建 .env 文件："
echo "     cp .env.example .env"
echo "  2. 编辑 .env 文件，设置："
echo "     DOUBAO_API_KEY=your_api_key"
echo "     LLM_PROVIDER=doubao"
echo ""
echo "方法三：添加到 ~/.zshrc（永久，所有终端）"
echo "  echo 'export DOUBAO_API_KEY=your_api_key' >> ~/.zshrc"
echo "  echo 'export LLM_PROVIDER=doubao' >> ~/.zshrc"
echo "  source ~/.zshrc"
echo ""

