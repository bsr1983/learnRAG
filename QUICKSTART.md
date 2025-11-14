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
# 复制环境变量模板（如果 .env.example 存在）
# cp .env.example .env

# 编辑 .env 文件，至少配置：
# OPENAI_API_KEY=your_key_here
```

### 3. 启动 Qdrant（向量数据库）

```bash
# 使用 Docker 启动 Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 验证 Qdrant 运行正常
curl http://localhost:6333/health
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

