# 验证配置参数已设置成功

## 快速验证方法

### 方法一：使用验证脚本（推荐）

```bash
# 运行验证脚本
python scripts/test_llm.py
```

如果看到 "✅ 配置验证成功！可以正常使用 LLM"，说明配置正确。

### 方法二：手动检查环境变量

```bash
# 检查环境变量
echo $DOUBAO_API_KEY
echo $LLM_PROVIDER

# 如果显示值，说明已设置
# 如果为空，说明未设置
```

### 方法三：使用 Python 检查

```python
import os
from dotenv import load_dotenv

load_dotenv()

print("DOUBAO_API_KEY:", "✅ 已设置" if os.getenv("DOUBAO_API_KEY") else "❌ 未设置")
print("LLM_PROVIDER:", os.getenv("LLM_PROVIDER", "❌ 未设置"))
```

## 常见问题

### Q: 为什么 export 后验证脚本还是显示未设置？

A: 可能的原因：
1. **在不同的终端中运行** - export 只在当前终端有效
2. **环境变量名称错误** - 检查大小写
3. **值中包含特殊字符** - 需要用引号包裹

**解决方案：**
```bash
# 方法1: 在同一终端中设置并运行
export DOUBAO_API_KEY=your_key
export LLM_PROVIDER=doubao
python scripts/test_llm.py

# 方法2: 使用 .env 文件（推荐）
echo 'DOUBAO_API_KEY=your_key' > .env
echo 'LLM_PROVIDER=doubao' >> .env
python scripts/test_llm.py
```

### Q: 如何确认 .env 文件被正确加载？

A: 运行验证脚本，如果显示"已设置"，说明 .env 文件被正确加载。

### Q: 验证脚本显示成功，但运行 RAG 系统还是报错？

A: 可能的原因：
1. **RAG 系统在另一个进程中运行** - 环境变量未传递
2. **IDE 运行** - IDE 可能没有加载环境变量

**解决方案：**
- 使用命令行运行：`python app/integrated_rag_system.py`
- 或使用提供的脚本：`bash scripts/run_rag.sh`

## 完整验证流程

```bash
# 1. 设置环境变量（或创建 .env 文件）
export DOUBAO_API_KEY=your_key
export LLM_PROVIDER=doubao

# 2. 验证环境变量
python scripts/test_llm.py

# 3. 如果验证成功，运行 RAG 系统
python app/integrated_rag_system.py
```

## 调试技巧

如果配置验证失败，检查：

1. **API Key 格式**
   - 不要包含引号
   - 不要有多余的空格
   - 确保是完整的 Key

2. **环境变量作用域**
   - `export` 只在当前 shell 有效
   - `.env` 文件会被 `python-dotenv` 自动加载
   - `~/.zshrc` 对所有新终端有效

3. **查看实际加载的值**
   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv()
   key = os.getenv("DOUBAO_API_KEY")
   print(f"Key length: {len(key) if key else 0}")
   print(f"Key prefix: {key[:10] if key else 'None'}...")
   ```

