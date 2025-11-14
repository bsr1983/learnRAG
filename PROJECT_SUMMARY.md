# 项目总结：RAG + 智能体编排系统学习路线

## 📖 项目简介

本项目是一个完整的 **RAG（Retrieval-Augmented Generation）系统学习路线**，旨在帮助学习者从零开始，在 3 周内深入理解 RAG 技术栈，掌握从嵌入模型到评测系统的完整流程，并具备自研编排框架的能力。

## 🎯 学习目标

通过系统学习，你将能够：

1. ✅ **深入理解 RAG 技术栈**
   - 嵌入模型（bge-large-zh, m3e-large）
   - 向量数据库（Qdrant）
   - 重排模型（bge-reranker-base）
   - RAG-Fusion 多查询融合
   - 结构化输出（DSPy, Guidance）
   - 系统评测（Ragas, ChatArena）

2. ✅ **独立构建 RAG 系统**
   - 从零开始构建完整的 RAG 系统
   - 优化系统性能
   - 进行系统评测

3. ✅ **理解编排框架**
   - LangChain / LlamaIndex 的设计思想
   - 自研编排框架的能力

## 📂 项目结构

```
learnRAG/
├── README.md                    # 主学习路线文档（详细）
├── QUICKSTART.md                # 快速开始指南
├── PROJECT_SUMMARY.md           # 项目总结（本文件）
├── requirements.txt             # Python 依赖
├── config.yaml                  # 配置文件
│
├── embeddings/                  # 嵌入模型模块
│   ├── embed_model.py          # bge-large-zh / m3e-large
│   └── reranker.py             # bge-reranker-base
│
├── storage/                     # 向量存储模块
│   └── qdrant_client.py        # Qdrant 客户端
│
├── retrieval/                   # 检索模块
│   ├── basic_rag_demo.py       # 基础 RAG
│   └── rag_fusion_demo.py       # RAG-Fusion
│
├── llm/                         # LLM 模块
│   └── structured_output_demo.py # 结构化输出
│
├── evaluation/                  # 评测模块
│   └── ragas_eval_demo.py      # Ragas 评测
│
├── app/                         # 应用入口
│   └── integrated_rag_system.py # 完整系统
│
├── docs/                        # 文档目录
│   ├── architecture.md         # 系统架构
│   ├── learning_checklist.md   # 学习检查清单
│   ├── learning_notes.md       # 学习笔记模板
│   ├── learning_suggestions.md # 学习建议
│   ├── references.md          # 参考资料汇总
│   └── execution_plan.md       # 执行计划
│
└── data/                        # 数据目录
    ├── documents/              # 测试文档
    └── evaluations/            # 评测数据集
```

## 📅 学习计划概览

### 第 1 周：RAG 系统核心组件入门

**目标**：掌握嵌入、向量检索、重排的基础能力

- **Day 1-2**: 语义嵌入与向量基础
- **Day 3-4**: 向量数据库
- **Day 5-7**: 构建最小 RAG Demo

### 第 2 周：增强与结构化生成

**目标**：掌握多查询融合和结构化输出技术

- **Day 8-10**: RAG-Fusion 与查询增强
- **Day 11-13**: 结构化输出与抽取
- **Day 14**: 系统整合

### 第 3 周：评测与编排体系建设

**目标**：掌握系统评测和编排框架设计

- **Day 15-17**: Ragas 检索评测
- **Day 18-19**: ChatArena 质量评测
- **Day 20-21**: 编排框架与自研准备

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动 Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 2. 运行第一个示例

```bash
# Day 1-2: 嵌入模型示例
python embeddings/basic_embedding_demo.py
```

### 3. 开始学习

按照 `README.md` 中的详细学习路线，从 Day 1 开始，循序渐进。

## 📚 核心文档

1. **README.md** - 主学习路线文档
   - 详细的学习计划（21 天）
   - 每天的学习目标、实践任务、参考资料
   - 项目结构说明

2. **QUICKSTART.md** - 快速开始指南
   - 5 分钟快速上手
   - 常见问题解答

3. **docs/learning_checklist.md** - 学习检查清单
   - 每天/每周的检查点
   - 进度追踪

4. **docs/learning_suggestions.md** - 学习建议
   - 学习原则和方法
   - 实践技巧

5. **docs/execution_plan.md** - 执行计划
   - 详细的执行步骤
   - 参考资料使用指南

6. **docs/references.md** - 参考资料汇总
   - 论文、文档、教程链接
   - 学习资源

## 🎓 学习建议

### 1. 按顺序学习
严格按照 Day 1 → Day 21 的顺序，不要跳跃。

### 2. 理论与实践结合
每学习一个概念，立即动手实践。

### 3. 记录与总结
每天记录学习笔记，每周进行复盘。

### 4. 实验与对比
多做实验，对比不同方法的效果。

## ✅ 学习检查

使用 `docs/learning_checklist.md` 追踪学习进度：

- 每天完成后打勾
- 每周统计完成率
- 及时发现问题并调整

## 📊 预期成果

完成本学习计划后，你将：

1. ✅ 拥有一个完整的 RAG 系统代码库
2. ✅ 深入理解 RAG 技术栈的每个组件
3. ✅ 具备独立构建和优化 RAG 系统的能力
4. ✅ 具备系统评测和性能优化的能力
5. ✅ 具备自研编排框架的基础能力

## 🔗 相关资源

- **论文**: 见 `docs/references.md`
- **文档**: 各框架的官方文档
- **代码**: 本项目的示例代码
- **社区**: LangChain、LlamaIndex 等社区

## 💡 下一步

1. **开始学习**: 阅读 `README.md`，从 Day 1 开始
2. **环境准备**: 按照 `QUICKSTART.md` 准备环境
3. **记录进度**: 使用 `docs/learning_checklist.md` 追踪进度
4. **寻求帮助**: 遇到问题查阅文档或参与社区讨论

---

**祝你学习顺利！🚀**

如有问题，请查阅相关文档或提交 Issue。

