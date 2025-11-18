# 安装问题排查指南

## 常见问题及解决方案

### 1. NumPy 版本兼容性问题

**问题**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**解决方案**:
```bash
pip install "numpy<2.0.0"
```

已在 `requirements.txt` 中固定为 `numpy>=1.24.0,<2.0.0`

### 2. Hugging Face 模型下载失败

**问题**: 无法连接到 `https://huggingface.co` 下载模型

**解决方案 A: 使用镜像源（推荐）**

```bash
# 临时使用（当前终端会话）
export HF_ENDPOINT=https://hf-mirror.com

# 永久配置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

**解决方案 B: 配置环境变量**

在运行 Python 脚本前设置：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python embeddings/basic_embedding_demo.py
```

**解决方案 C: 在代码中设置**

在 Python 代码开头添加：
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**解决方案 D: 手动下载模型**

1. 访问 https://hf-mirror.com/BAAI/bge-large-zh
2. 下载所有文件到本地目录，例如 `./models/bge-large-zh/`
3. 在代码中使用本地路径：
```python
embedder = EmbeddingModel(model_name="./models/bge-large-zh")
```

### 3. Docker 连接问题

**问题**: `Connection reset by peer` 或无法拉取镜像

**解决方案**:
- 检查 Docker Desktop 是否正常运行
- 检查网络连接
- 使用镜像安装方案（见 QUICKSTART.md）

### 4. PyTorch 安装问题

**问题**: Python 3.13+ 无法安装 PyTorch

**解决方案**: 使用 Python 3.11 或 3.12 创建虚拟环境：
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. 模型加载内存不足

**问题**: 内存不足导致模型加载失败

**解决方案**:
- 使用更小的模型（如 `all-MiniLM-L6-v2`）
- 设置 `device="cpu"`（如果使用 GPU）
- 减少批处理大小

## 推荐的镜像源配置

### Hugging Face 镜像

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export HF_ENDPOINT=https://hf-mirror.com
```

### Python 包镜像（可选）

```bash
# 使用清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 验证安装

运行以下命令验证环境：

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查关键包版本
python -c "import torch; import numpy; import sentence_transformers; print(f'PyTorch: {torch.__version__}'); print(f'NumPy: {numpy.__version__}'); print('✅ 所有包已正确安装')"

# 测试模型下载（如果配置了镜像）
export HF_ENDPOINT=https://hf-mirror.com
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('✅ 模型下载成功')"
```

## 获取帮助

如果以上方案都无法解决问题，请：
1. 检查网络连接
2. 查看详细错误信息
3. 查阅项目文档
4. 提交 Issue（附上错误日志）

