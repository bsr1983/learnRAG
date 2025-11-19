#!/bin/bash
# 运行 RAG 系统的便捷脚本
# 自动检查环境变量并运行

cd "$(dirname "$0")/.."

echo "=========================================="
echo "RAG 系统运行脚本"
echo "=========================================="
echo ""

# 检查环境变量
if [ -z "$DOUBAO_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: 未检测到 LLM API Key"
    echo ""
    echo "请设置环境变量:"
    echo "  export DOUBAO_API_KEY=your_key"
    echo "  export LLM_PROVIDER=doubao"
    echo ""
    echo "或者创建 .env 文件:"
    echo "  echo 'DOUBAO_API_KEY=your_key' > .env"
    echo "  echo 'LLM_PROVIDER=doubao' >> .env"
    echo ""
    read -p "是否继续运行? (y/n): " continue_run
    if [ "$continue_run" != "y" ]; then
        exit 1
    fi
fi

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ 错误: 虚拟环境不存在，请先运行: python -m venv venv"
    exit 1
fi

# 设置其他环境变量
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

# 运行脚本
echo "正在运行 RAG 系统..."
echo ""
python app/integrated_rag_system.py

