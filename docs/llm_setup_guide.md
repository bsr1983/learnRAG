# LLM 配置完整指南

## 问题：环境变量已设置但系统仍无法使用

如果您已经执行了 `export DOUBAO_API_KEY=...` 但系统仍然报错，可能是因为：

1. **环境变量只在当前终端会话有效**
2. **Python 脚本在另一个 shell 中运行**
3. **需要使用 .env 文件来持久化配置**

## 解决方案

### 方法一：在当前终端设置并运行（临时测试）

```bash
# 1. 设置环境变量（在同一终端中）
export DOUBAO_API_KEY=your_actual_api_key_here
export LLM_PROVIDER=doubao

# 2. 验证设置
echo $DOUBAO_API_KEY
echo $LLM_PROVIDER

# 3. 在同一终端中运行脚本
cd /Users/mac/workspaceForLearn/learnRAG
source venv/bin/activate
python scripts/test_llm.py
```

### 方法二：使用 .env 文件（推荐，永久配置）

```bash
# 1. 创建 .env 文件
cd /Users/mac/workspaceForLearn/learnRAG
cat > .env << EOF
DOUBAO_API_KEY=your_actual_api_key_here
LLM_PROVIDER=doubao
HF_ENDPOINT=https://hf-mirror.com
EOF

# 2. 验证配置
python scripts/test_llm.py
```

**注意**：`.env` 文件会被自动加载，无需手动 export。

### 方法三：添加到 ~/.zshrc（永久，所有终端）

```bash
# 1. 添加到配置文件
echo 'export DOUBAO_API_KEY=your_actual_api_key_here' >> ~/.zshrc
echo 'export LLM_PROVIDER=doubao' >> ~/.zshrc

# 2. 重新加载配置
source ~/.zshrc

# 3. 验证
echo $DOUBAO_API_KEY

# 4. 测试
cd /Users/mac/workspaceForLearn/learnRAG
source venv/bin/activate
python scripts/test_llm.py
```

## 验证配置

### 快速验证脚本

```bash
# 运行验证脚本
python scripts/test_llm.py
```

### 手动验证

```python
import os
from dotenv import load_dotenv

load_dotenv()

print("DOUBAO_API_KEY:", "已设置" if os.getenv("DOUBAO_API_KEY") else "未设置")
print("LLM_PROVIDER:", os.getenv("LLM_PROVIDER", "未设置"))

# 测试 LLM
from llm.llm_client import get_llm_client
client = get_llm_client(provider="doubao")
response = client.generate("你好")
print("LLM 响应:", response)
```

## 常见问题

### Q: 为什么 export 后还是不行？

A: `export` 只在当前终端会话有效。如果：
- 在终端 A 中执行 `export`
- 在终端 B 中运行 Python 脚本
- 或者在 IDE 中运行脚本

那么环境变量不会生效。**推荐使用 .env 文件**。

### Q: .env 文件在哪里？

A: `.env` 文件应该在项目根目录：
```
/Users/mac/workspaceForLearn/learnRAG/.env
```

### Q: 如何检查 .env 文件是否被加载？

A: 运行验证脚本：
```bash
python scripts/test_llm.py
```

如果显示"已设置"，说明 .env 文件被正确加载。

### Q: API Key 格式是什么？

A: 豆包的 API Key 通常是：
- 格式：`sk-xxxxx...` 或类似格式
- 长度：通常 30-50 个字符
- **不要包含引号**：`DOUBAO_API_KEY=sk-xxx` ✅ 正确
- **不要包含引号**：`DOUBAO_API_KEY="sk-xxx"` ❌ 错误

### Q: 如何获取豆包 API Key？

A: 
1. 访问 https://console.volcengine.com/
2. 注册/登录账号
3. 开通豆包服务
4. 在控制台获取 API Key

## 推荐配置方式

**对于学习和开发，推荐使用 .env 文件**：

1. 创建 `.env` 文件
2. 添加配置
3. 添加到 `.gitignore`（不要提交 API Key）
4. 代码会自动加载

这样配置：
- ✅ 永久有效
- ✅ 不需要每次 export
- ✅ 可以在 IDE 中直接运行
- ✅ 不会影响其他项目

