# 快速开始指南

## 🚀 5 分钟快速上手

### 1. 环境准备

```bash
# 克隆或进入项目目录
cd learnRAG

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，配置 LLM API Key
# 推荐使用豆包（中文支持好，价格便宜）：
# DOUBAO_API_KEY=your_doubao_api_key
# LLM_PROVIDER=doubao

# 或者使用 OpenAI：
# OPENAI_API_KEY=your_openai_key
# LLM_PROVIDER=openai

# 支持的提供商：doubao, openai, qwen, ernie, zhipu
# 详细配置说明请查看 docs/llm_providers.md
```

### 3. 启动 Qdrant（向量数据库）

#### 方式一：直接运行（需要网络连接）

```bash
# 使用 Docker 启动 Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 验证 Qdrant 运行正常
curl http://localhost:6333/health
```

#### 方式二：手动通过镜像安装（离线/手动安装）

如果您需要手动安装或网络连接不稳定，可以按以下步骤操作：

> 💡 **提示**: 也可以使用提供的安装脚本快速完成安装：
> ```bash
> ./scripts/setup_qdrant.sh
> ```

**步骤 1: 下载镜像文件**

在有网络的环境中，先下载 Qdrant 镜像并保存为 tar 文件：

```bash
# 拉取 Qdrant 镜像
docker pull qdrant/qdrant:latest

# 将镜像保存为 tar 文件（方便传输和备份）
docker save qdrant/qdrant:latest -o qdrant-image.tar

# 或者指定版本（推荐）
docker pull qdrant/qdrant:v1.7.4
docker save qdrant/qdrant:v1.7.4 -o qdrant-image-v1.7.4.tar
```

**步骤 2: 加载镜像到 Docker**

将镜像文件传输到目标机器后，加载镜像：

```bash
# 加载镜像文件
docker load -i qdrant-image.tar

# 或者如果使用版本号
docker load -i qdrant-image-v1.7.4.tar

# 验证镜像已加载
docker images | grep qdrant
```

**步骤 3: 运行 Qdrant 容器**

```bash
# 运行 Qdrant 容器（前台运行，可以看到日志）
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 或者后台运行（推荐）
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 查看容器状态
docker ps | grep qdrant

# 查看日志
docker logs qdrant

# 停止容器
docker stop qdrant

# 启动已停止的容器
docker start qdrant

# 删除容器（注意：会删除数据，除非使用了数据卷）
docker rm qdrant
```

**步骤 4: 验证安装**

```bash
# 检查健康状态
curl http://localhost:6333/healthz

# 或者使用浏览器访问
# http://localhost:6333/dashboard
```

**常用 Docker 命令参考**

```bash
# 查看所有容器（包括已停止的）
docker ps -a

# 查看容器日志
docker logs -f qdrant

# 进入容器内部
docker exec -it qdrant sh

# 查看容器资源使用情况
docker stats qdrant

# 备份数据卷（如果使用了数据卷）
docker run --rm -v qdrant_storage:/data -v $(pwd):/backup \
  alpine tar czf /backup/qdrant-backup.tar.gz /data
```

### 4. 运行第一个示例

```bash
# Day 1-2: 嵌入模型示例
python embeddings/basic_embedding_demo.py

# Day 3-4: 向量数据库示例
python storage/qdrant_demo.py

# Day 5-7: 基础 RAG 示例
python retrieval/basic_rag_demo.py
```

## 📚 学习路径

### 第 1 周：基础组件
1. **Day 1-2**: 学习嵌入模型 → 运行 `embeddings/basic_embedding_demo.py`
2. **Day 3-4**: 学习向量数据库 → 运行 `storage/qdrant_demo.py`
3. **Day 5-7**: 构建 RAG 系统 → 运行 `retrieval/basic_rag_demo.py`

### 第 2 周：增强功能
1. **Day 8-10**: RAG-Fusion → 运行 `retrieval/rag_fusion_demo.py`
2. **Day 11-13**: 结构化输出 → 运行 `llm/structured_output_demo.py`
3. **Day 14**: 系统整合 → 运行 `app/integrated_rag_system.py`

### 第 3 周：评测与编排
1. **Day 15-17**: Ragas 评测 → 运行 `evaluation/ragas_eval_demo.py`
2. **Day 18-19**: ChatArena 评测 → 学习对话评测
3. **Day 20-21**: 自研编排器 → 设计并实现编排框架

## 🔧 常见问题

### Q: Qdrant 启动失败？
A: 确保 Docker 正在运行，端口 6333 未被占用

### Q: 模型下载慢？
A: 可以使用镜像源或手动下载模型到本地

### Q: OpenAI API 调用失败？
A: 检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确配置

### Q: 内存不足？
A: 可以：
- 使用 CPU 模式（修改代码中的 device="cpu"）
- 减少批处理大小
- 使用更小的模型

## 📖 详细文档

- **完整学习路线**: 查看 [README.md](README.md)
- **系统架构**: 查看 [docs/architecture.md](docs/architecture.md)
- **学习检查清单**: 查看 [docs/learning_checklist.md](docs/learning_checklist.md)
- **参考资料**: 查看 [docs/references.md](docs/references.md)

## 💡 学习建议

1. **按顺序学习**: 从 Day 1 开始，循序渐进
2. **动手实践**: 不要只看代码，要运行并修改
3. **记录笔记**: 使用 `docs/learning_notes.md` 记录学习过程
4. **遇到问题**: 先查阅文档，再查看 GitHub Issues

---

**开始你的 RAG 学习之旅吧！🎉**

