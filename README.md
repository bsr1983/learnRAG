# RAG + 智能体编排系统实战学习路线图

> 深入理解 RAG 技术栈，从零到一构建完整的检索增强生成系统

## 📋 目录

- [学习目标](#学习目标)
- [前置要求](#前置要求)
- [三周学习计划](#三周学习计划)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [学习检查清单](#学习检查清单)

---

## 🎯 学习目标

完成本路线后，你将能够：

1. ✅ 理解 RAG 系统的核心原理和架构设计
2. ✅ 掌握嵌入模型、向量数据库、重排模型的使用
3. ✅ 实现 RAG-Fusion 多查询融合技术
4. ✅ 使用 DSPy/Guidance 进行结构化输出
5. ✅ 使用 Ragas 进行系统评测
6. ✅ 理解 LangChain/LlamaIndex 编排框架
7. ✅ 具备自研编排层的基础能力

---

## 📚 前置要求

### 基础知识
- Python 3.9+ 编程基础
- 基本的机器学习概念（向量、相似度、嵌入）
- 了解 REST API 和 Docker 基础操作

### 环境准备
```bash
# Python 环境
python --version  # 确保 >= 3.9

# Docker（用于运行 Qdrant）
docker --version

# Git
git --version
```

---

## 📅 三周学习计划

### 第 1 周：RAG 系统核心组件入门

#### **Day 1-2: 语义嵌入与向量基础**

**学习目标：**
- 理解什么是 embedding 向量和语义相似度
- 掌握 dual-encoder vs cross-encoder 的区别
- 能够使用 bge-large-zh 和 m3e-large 生成句向量

**可执行步骤：**

1. **理论学习（2小时）**
   - [ ] 阅读：什么是词向量/句向量（Word2Vec → BERT → Sentence-BERT）
   - [ ] 理解：余弦相似度、欧氏距离、点积相似度
   - [ ] 理解：dual-encoder（双塔）vs cross-encoder（交叉编码器）架构差异

2. **实践任务（4小时）**
   - [ ] 安装 Sentence Transformers 库
   - [ ] 使用 bge-large-zh 生成中文句向量
   - [ ] 使用 m3e-large 测试中英双语相似度
   - [ ] 完成 `embeddings/basic_embedding_demo.py` 示例

3. **小实验（2小时）**
   - [ ] 输入 5 段不同文本，计算两两相似度
   - [ ] 可视化相似度矩阵（使用 matplotlib/seaborn）
   - [ ] 分析：为什么某些文本相似度高/低？

**参考资料：**

- **论文与理论：**
  - [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - 理解 Sentence-BERT 原理
  - [BGE: BAAI General Embedding](https://arxiv.org/abs/2309.07597) - BGE 模型论文
  - [Dual-encoder vs Cross-encoder 详解](https://www.sbert.net/examples/applications/cross-encoder/README.html)

- **代码与文档：**
  - [Sentence Transformers 官方文档](https://www.sbert.net/)
  - [HuggingFace Sentence Transformers 教程](https://huggingface.co/docs/transformers/main/en/model_doc/sentence-transformers)
  - [BAAI/bge-large-zh 模型卡片](https://huggingface.co/BAAI/bge-large-zh)
  - [moka-ai/m3e-large 模型卡片](https://huggingface.co/moka-ai/m3e-large)

- **实践教程：**
  - [Sentence Transformers 快速开始](https://www.sbert.net/docs/quickstart.html)
  - [BGE 模型使用指南](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

**检查点：**
- [ ] 能够解释 embedding 向量的含义
- [ ] 能够独立使用 bge-large-zh 生成向量
- [ ] 理解为什么需要重排模型（cross-encoder）

---

#### **Day 3-4: 向量数据库**

**学习目标：**
- 理解向量索引算法（HNSW、IVF、PQ）
- 掌握 Qdrant 的基本操作
- 能够将嵌入向量存储到向量库并进行检索

**可执行步骤：**

1. **理论学习（2小时）**
   - [ ] 理解：为什么需要向量数据库（vs 传统数据库）
   - [ ] 学习：HNSW（Hierarchical Navigable Small World）算法原理
   - [ ] 了解：IVF（Inverted File Index）和 PQ（Product Quantization）
   - [ ] 理解：近似最近邻搜索（ANN）vs 精确最近邻（KNN）

2. **环境搭建（1小时）**
   - [ ] 使用 Docker 启动 Qdrant 服务
   - [ ] 验证 Qdrant 服务正常运行
   - [ ] 安装 Python Qdrant 客户端

3. **实践任务（5小时）**
   - [ ] 创建 Qdrant 集合（Collection）
   - [ ] 批量插入嵌入向量（使用 bge-large-zh）
   - [ ] 实现向量检索功能（相似度搜索）
   - [ ] 完成 `storage/qdrant_demo.py` 示例
   - [ ] 对比不同索引参数的效果（HNSW m、ef_construct）

4. **进阶实验（2小时）**
   - [ ] 实现批量插入和检索的性能测试
   - [ ] 测试不同向量维度对性能的影响
   - [ ] 了解 Qdrant 的过滤功能（metadata filtering）

**参考资料：**

- **算法原理：**
  - [HNSW 论文](https://arxiv.org/abs/1603.09320) - Efficient and robust approximate nearest neighbor search
  - [IVF 和 PQ 算法详解](https://github.com/facebookresearch/faiss/wiki) - Faiss 文档
  - [向量数据库技术综述](https://www.pinecone.io/learn/vector-database/)

- **Qdrant 文档：**
  - [Qdrant 官方文档](https://qdrant.tech/documentation/)
  - [Qdrant Python 客户端](https://qdrant.github.io/qdrant-client/)
  - [Qdrant 快速开始](https://qdrant.tech/documentation/quick-start/)
  - [Qdrant Docker 部署](https://qdrant.tech/documentation/guides/installation/)

- **Milvus 对比学习（可选）：**
  - [Milvus 官方文档](https://milvus.io/docs)
  - [Qdrant vs Milvus 对比](https://www.qdrant.tech/compare/milvus)

- **实践教程：**
  - [Qdrant Python SDK 示例](https://github.com/qdrant/qdrant-client/tree/master/examples)
  - [向量数据库性能优化指南](https://qdrant.tech/documentation/tutorials/optimize-performance/)

**检查点：**
- [ ] 能够解释 HNSW 算法的基本思想
- [ ] 能够独立使用 Qdrant 存储和检索向量
- [ ] 理解向量索引参数对性能的影响

---

#### **Day 5-7: 构建最小 RAG Demo**

**学习目标：**
- 理解 RAG 的完整流程：检索 → 拼接 → 生成
- 使用 LangChain 构建 RetrievalQA 系统
- 集成重排模型提升检索质量

**可执行步骤：**

1. **理论学习（2小时）**
   - [ ] 阅读 RAG 原始论文：["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401)
   - [ ] 理解 RAG 的两种模式：RAG-Sequence 和 RAG-Token
   - [ ] 理解为什么需要重排（reranking）步骤

2. **LangChain 入门（2小时）**
   - [ ] 学习 LangChain 核心概念：LLM、PromptTemplate、Retriever、Chain
   - [ ] 完成 LangChain 官方快速开始教程
   - [ ] 理解 LangChain 的模块化设计思想

3. **实践任务（8小时）**
   - [ ] 准备测试文档集（至少 10 篇文档）
   - [ ] 使用 LangChain 的 TextSplitter 进行文档分块
   - [ ] 构建向量存储（使用 Qdrant + bge-large-zh）
   - [ ] 实现基础 RAG 流程：问题 → 检索 → LLM 回答
   - [ ] 集成 bge-reranker-base 进行结果重排
   - [ ] 完成 `retrieval/basic_rag_demo.py` 示例
   - [ ] 对比：无重排 vs 有重排的准确度差异

4. **优化实验（4小时）**
   - [ ] 测试不同的 chunk_size 和 chunk_overlap 参数
   - [ ] 实现 top-k 检索 + 重排的完整流程
   - [ ] 记录检索到的文档和最终答案，分析质量

**参考资料：**

- **RAG 理论：**
  - [RAG 原始论文](https://arxiv.org/abs/2005.11401) - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
  - [RAG 综述论文](https://arxiv.org/abs/2312.10997) - Retrieval-Augmented Generation: A Survey
  - [LangChain RAG 教程](https://python.langchain.com/docs/use_cases/question_answering/)

- **LangChain 文档：**
  - [LangChain 官方文档](https://python.langchain.com/)
  - [LangChain 快速开始](https://python.langchain.com/docs/get_started/introduction)
  - [LangChain RetrievalQA 示例](https://python.langchain.com/docs/use_cases/question_answering/)
  - [LangChain Vector Stores](https://python.langchain.com/docs/integrations/vectorstores/)

- **重排模型：**
  - [BAAI/bge-reranker-base 模型卡片](https://huggingface.co/BAAI/bge-reranker-base)
  - [Cross-encoder 重排原理](https://www.sbert.net/examples/applications/cross-encoder/README.html)
  - [LangChain Reranker 集成](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/)

- **实践教程：**
  - [LangChain RAG 完整示例](https://github.com/langchain-ai/langchain/tree/master/templates/rag-pinecone)
  - [文档分块最佳实践](https://www.pinecone.io/learn/chunking-strategies/)

**检查点：**
- [ ] 能够解释 RAG 的完整流程
- [ ] 能够独立构建一个可运行的 RAG 系统
- [ ] 理解重排模型如何提升检索质量
- [ ] 完成至少 10 个问题的测试，准确率 > 70%

---

### 第 2 周：增强与结构化生成

#### **Day 8-10: RAG-Fusion 与查询增强**

**学习目标：**
- 理解多查询融合的核心思想
- 实现 RAG-Fusion 技术
- 对比单查询 vs 多查询的召回质量

**可执行步骤：**

1. **理论学习（3小时）**
   - [ ] 阅读 RAG-Fusion 论文：["RAG-Fusion: Answering Ambiguous Queries with Query Ensemble"](https://arxiv.org/abs/2401.10415)
   - [ ] 理解查询改写的必要性（为什么需要多查询）
   - [ ] 学习查询融合策略：RRF（Reciprocal Rank Fusion）、加权平均等

2. **LangChain MultiQueryRetriever 学习（2小时）**
   - [ ] 阅读 LangChain MultiQueryRetriever 源码
   - [ ] 理解查询改写的 prompt 设计
   - [ ] 学习如何合并多个检索结果

3. **实践任务（10小时）**
   - [ ] 实现查询改写功能（使用 LLM 生成 3-5 个改写查询）
   - [ ] 实现多查询检索（对每个改写查询进行向量检索）
   - [ ] 实现结果融合（RRF 算法）
   - [ ] 完成 `retrieval/rag_fusion_demo.py` 示例
   - [ ] 对比实验：单查询 RAG vs RAG-Fusion（召回率、准确率）

4. **进阶优化（5小时）**
   - [ ] 实现查询去重（合并相似查询）
   - [ ] 实现动态查询数量（根据问题复杂度调整）
   - [ ] 测试不同融合策略的效果（RRF、加权平均、取并集）

**参考资料：**

- **RAG-Fusion 理论：**
  - [RAG-Fusion 论文](https://arxiv.org/abs/2401.10415) - Answering Ambiguous Queries with Query Ensemble
  - [Multi-Query RAG 技术详解](https://www.pinecone.io/learn/query-rewriting/)
  - [Reciprocal Rank Fusion 算法](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

- **LangChain 实现：**
  - [LangChain MultiQueryRetriever](https://python.langchain.com/docs/integrations/retrievers/multi-query-retriever/)
  - [LangChain Parent Document Retriever](https://python.langchain.com/docs/integrations/retrievers/parent-document-retriever/)
  - [LangChain Ensemble Retriever](https://python.langchain.com/docs/integrations/retrievers/ensemble-retriever/)

- **查询改写技术：**
  - [Query Expansion Techniques](https://en.wikipedia.org/wiki/Query_expansion)
  - [Query Rewriting for RAG](https://blog.langchain.dev/query-construction/)

- **实践教程：**
  - [RAG-Fusion 实现示例](https://github.com/Raudaschl/rag-fusion)
  - [多查询 RAG 实战](https://www.youtube.com/watch?v=example)

**检查点：**
- [ ] 能够解释 RAG-Fusion 的核心思想
- [ ] 能够独立实现多查询改写和融合
- [ ] 通过实验验证 RAG-Fusion 提升召回质量

---

#### **Day 11-13: 结构化输出与抽取**

**学习目标：**
- 理解结构化生成的概念和必要性
- 掌握 DSPy 的 Signature 和 Pipeline 用法
- 能够使用 Guidance 进行格式约束

**可执行步骤：**

1. **理论学习（3小时）**
   - [ ] 理解：为什么需要结构化输出（vs 自由文本）
   - [ ] 学习：约束生成（Constrained Generation）的概念
   - [ ] 了解：JSON Schema、Pydantic 模型在 LLM 中的应用

2. **DSPy 学习（5小时）**
   - [ ] 阅读 DSPy 论文：["DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"](https://arxiv.org/abs/2310.03714)
   - [ ] 学习 DSPy 核心概念：Signature、Module、Optimizer
   - [ ] 完成 DSPy 官方教程
   - [ ] 实现一个简单的 DSPy Pipeline：文本 → 结构化 JSON

3. **Guidance 学习（4小时）**
   - [ ] 学习 Guidance 模板语法
   - [ ] 理解 Guidance 如何控制 LLM 输出格式
   - [ ] 实现一个 Guidance 示例：约束输出为 JSON 格式

4. **实践任务（8小时）**
   - [ ] 使用 DSPy 从文本中抽取实体信息（人物、地点、时间）
   - [ ] 使用 Guidance 约束输出为特定 JSON Schema
   - [ ] 集成到 RAG 流程：RAG 检索 → DSPy 格式化输出
   - [ ] 完成 `llm/structured_output_demo.py` 示例
   - [ ] 对比：无约束 vs 有约束的输出质量

5. **进阶应用（4小时）**
   - [ ] 实现多字段抽取（从一段文本中提取多个结构化字段）
   - [ ] 实现嵌套结构抽取（JSON 嵌套对象）
   - [ ] 测试不同 LLM 的结构化输出能力

**参考资料：**

- **DSPy 理论：**
  - [DSPy 论文](https://arxiv.org/abs/2310.03714) - Compiling Declarative Language Model Calls
  - [DSPy 官方文档](https://dspy-docs.vercel.app/)
  - [DSPy GitHub](https://github.com/stanfordnlp/dspy)
  - [DSPy 教程](https://dspy-docs.vercel.app/docs/tutorials/intro)

- **Guidance 理论：**
  - [Microsoft Guidance GitHub](https://github.com/microsoft/guidance)
  - [Guidance 文档](https://guidance.readthedocs.io/)
  - [Guidance 快速开始](https://github.com/microsoft/guidance#quick-start)

- **结构化输出技术：**
  - [JSON Schema 约束生成](https://json-schema.org/)
  - [Pydantic 模型](https://docs.pydantic.dev/)
  - [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/structured/)

- **实践教程：**
  - [DSPy 示例集合](https://github.com/stanfordnlp/dspy/tree/main/examples)
  - [Guidance 示例](https://github.com/microsoft/guidance/tree/main/examples)
  - [结构化输出最佳实践](https://www.promptingguide.ai/techniques/structured_outputs)

**检查点：**
- [ ] 能够解释结构化输出的必要性
- [ ] 能够使用 DSPy 构建简单的 Pipeline
- [ ] 能够使用 Guidance 约束输出格式
- [ ] 完成至少 3 个结构化抽取任务

---

#### **Day 14: 本周复盘**

**学习目标：**
- 整合本周所学技术
- 构建完整的 RAG + RAG-Fusion + DSPy 流程
- 总结性能瓶颈和优化方向

**可执行步骤：**

1. **系统整合（4小时）**
   - [ ] 整合 RAG-Fusion 和 DSPy 到统一流程
   - [ ] 编写配置文件（YAML）管理参数
   - [ ] 完成 `app/integrated_rag_system.py` 主程序

2. **性能分析（2小时）**
   - [ ] 记录各模块耗时（检索、重排、生成）
   - [ ] 分析召回质量瓶颈
   - [ ] 总结优化方向

3. **文档整理（2小时）**
   - [ ] 编写系统架构文档
   - [ ] 记录遇到的问题和解决方案
   - [ ] 更新学习笔记

**检查点：**
- [ ] 完成一个端到端的 RAG 系统
- [ ] 能够清晰描述系统架构
- [ ] 识别至少 3 个优化点

---

### 第 3 周：评测与编排体系建设

#### **Day 15-17: Ragas 检索评测**

**学习目标：**
- 理解 RAG 系统的评测指标
- 使用 Ragas 进行自动化评测
- 能够分析评测报告并优化系统

**可执行步骤：**

1. **理论学习（3小时）**
   - [ ] 理解 RAG 评测的核心指标：
     - Faithfulness（忠实度）：答案是否基于检索到的上下文
     - Context Precision（上下文精确度）：检索到的上下文是否相关
     - Context Recall（上下文召回率）：是否检索到了所有相关信息
     - Answer Relevance（答案相关性）：答案是否回答了问题
   - [ ] 学习 Ragas 的评测框架设计

2. **Ragas 实践（6小时）**
   - [ ] 安装和配置 Ragas
   - [ ] 准备评测数据集（至少 20 个问答对）
   - [ ] 运行 Ragas 评测脚本
   - [ ] 分析评测报告，识别问题
   - [ ] 完成 `evaluation/ragas_eval_demo.py` 示例

3. **优化实验（5小时）**
   - [ ] 根据评测结果调整检索参数（top-k、chunk_size）
   - [ ] 对比不同重排策略的效果
   - [ ] 实现自动化评测流程（CI/CD 集成）

4. **自定义指标（4小时）**
   - [ ] 实现自定义评测指标（如领域特定指标）
   - [ ] 集成到 Ragas 评测流程
   - [ ] 生成可视化评测报告

**参考资料：**

- **Ragas 理论：**
  - [Ragas 论文](https://arxiv.org/abs/2309.15217) - RAGAS: Automated Evaluation of Retrieval Augmented Generation
  - [Ragas GitHub](https://github.com/explodinggradients/ragas)
  - [Ragas 官方文档](https://docs.ragas.io/)
  - [Ragas 快速开始](https://docs.ragas.io/get-started/quickstart)

- **RAG 评测理论：**
  - [RAG 评测综述](https://arxiv.org/abs/2312.10997) - Retrieval-Augmented Generation: A Survey
  - [RAG 评测指标详解](https://www.pinecone.io/learn/rag-evaluation/)
  - [LLM 评测方法](https://arxiv.org/abs/2303.16634) - Evaluating Large Language Models

- **实践教程：**
  - [Ragas 示例集合](https://github.com/explodinggradients/ragas/tree/main/examples)
  - [RAG 评测最佳实践](https://docs.ragas.io/concepts/metrics)
  - [构建 RAG 评测数据集](https://docs.ragas.io/concepts/datasets)

**检查点：**
- [ ] 能够解释 RAG 评测的核心指标
- [ ] 能够使用 Ragas 进行自动化评测
- [ ] 能够根据评测结果优化系统
- [ ] 完成至少 20 个问答对的评测

---

#### **Day 18-19: ChatArena 质量评测**

**学习目标：**
- 理解对话系统的评测方法
- 使用 ChatArena 进行多模型对比评测
- 掌握 Pairwise LLM Evaluation 思路

**可执行步骤：**

1. **理论学习（2小时）**
   - [ ] 理解对话系统评测的挑战
   - [ ] 学习 Pairwise Comparison（成对比较）方法
   - [ ] 了解 ArenaBench 评测框架

2. **ChatArena 实践（6小时）**
   - [ ] 安装和配置 ChatArena
   - [ ] 准备评测问题集
   - [ ] 实现两个版本的 RAG 系统（基础版 vs Fusion 版）
   - [ ] 运行 ChatArena 对比评测
   - [ ] 分析评测结果

3. **进阶实验（4小时）**
   - [ ] 实现多轮对话评测
   - [ ] 测试不同 LLM 的对话质量
   - [ ] 实现自动化评测报告生成

**参考资料：**

- **ChatArena 理论：**
  - [ChatArena GitHub](https://github.com/chatarena/chatarena)
  - [ArenaBench 论文](https://arxiv.org/abs/2402.05668) - ArenaBench: A Benchmark for Arena Evaluation of Language Models
  - [LLM 对话评测方法](https://arxiv.org/abs/2308.01320) - Evaluating Large Language Models

- **对话评测理论：**
  - [对话系统评测综述](https://arxiv.org/abs/2006.14711) - Evaluating Dialogue Systems
  - [Pairwise Comparison 方法](https://en.wikipedia.org/wiki/Pairwise_comparison)

**检查点：**
- [ ] 能够使用 ChatArena 进行对话评测
- [ ] 理解 Pairwise Comparison 方法
- [ ] 完成至少 10 个问题的对比评测

---

#### **Day 20-21: 编排框架与自研准备**

**学习目标：**
- 深入理解 LangChain 和 LlamaIndex 的架构设计
- 设计自研编排框架的思路
- 实现一个最小化的自研编排器

**可执行步骤：**

1. **框架对比学习（4小时）**
   - [ ] 阅读 LangChain 核心源码（Chain、Agent、Memory）
   - [ ] 阅读 LlamaIndex 核心源码（QueryEngine、Retriever）
   - [ ] 对比两种框架的设计理念和优缺点
   - [ ] 总结编排框架的核心组件

2. **自研设计（4小时）**
   - [ ] 设计自研编排框架的架构图
   - [ ] 定义核心接口（Task、Pipeline、Executor）
   - [ ] 设计任务调度机制（顺序、并行、条件分支）
   - [ ] 设计内存管理机制（对话历史、上下文管理）

3. **实现最小编排器（8小时）**
   - [ ] 实现基础 Pipeline 类（支持链式调用）
   - [ ] 实现 Task 抽象类（检索、重排、生成）
   - [ ] 实现 Executor 执行器（任务调度）
   - [ ] 完成 `orchestration/minimal_orchestrator.py` 示例
   - [ ] 测试：用自研编排器重构之前的 RAG 流程

4. **优化与扩展（4小时）**
   - [ ] 实现错误处理和重试机制
   - [ ] 实现日志和监控功能
   - [ ] 实现配置管理（YAML/JSON）
   - [ ] 性能对比：自研 vs LangChain

**参考资料：**

- **LangChain 源码：**
  - [LangChain GitHub](https://github.com/langchain-ai/langchain)
  - [LangChain 架构文档](https://python.langchain.com/docs/architecture/)
  - [LangChain Chain 源码](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/chains)

- **LlamaIndex 源码：**
  - [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
  - [LlamaIndex 架构文档](https://docs.llamaindex.ai/en/stable/core_modules/)
  - [LlamaIndex QueryEngine 源码](https://github.com/run-llama/llama_index/tree/main/llama_index/core/query_engine)

- **编排框架设计：**
  - [工作流引擎设计模式](https://www.workflowpatterns.com/)
  - [事件驱动架构](https://martinfowler.com/articles/201701-event-driven.html)
  - [任务编排最佳实践](https://www.conductor.com/nightlight/orchestration-vs-choreography/)

**检查点：**
- [ ] 能够解释 LangChain 和 LlamaIndex 的架构差异
- [ ] 能够设计自研编排框架的架构
- [ ] 完成一个最小化的自研编排器
- [ ] 能够用自研编排器运行 RAG 流程

---

## 📂 项目结构

```
learnRAG/
├── README.md                    # 本文件：学习路线图
├── requirements.txt             # Python 依赖
├── config.yaml                  # 配置文件
├── .env.example                 # 环境变量示例
│
├── embeddings/                  # 嵌入模型模块
│   ├── __init__.py
│   ├── embed_model.py          # bge-large-zh / m3e-large 封装
│   ├── reranker.py             # bge-reranker-base 封装
│   └── basic_embedding_demo.py # Day 1-2 示例
│
├── storage/                     # 向量存储模块
│   ├── __init__.py
│   ├── qdrant_client.py        # Qdrant 客户端封装
│   ├── index_builder.py        # 索引构建工具
│   └── qdrant_demo.py          # Day 3-4 示例
│
├── retrieval/                   # 检索模块
│   ├── __init__.py
│   ├── basic_rag_demo.py       # Day 5-7 基础 RAG
│   ├── rag_fusion_demo.py      # Day 8-10 RAG-Fusion
│   └── retriever.py            # 检索器封装
│
├── llm/                         # LLM 模块
│   ├── __init__.py
│   ├── generator.py            # LLM 生成器
│   ├── dsp_prompt.py           # DSPy 集成
│   ├── guidance_prompt.py      # Guidance 集成
│   └── structured_output_demo.py # Day 11-13 示例
│
├── evaluation/                  # 评测模块
│   ├── __init__.py
│   ├── ragas_eval.py           # Ragas 评测
│   ├── ragas_eval_demo.py      # Day 15-17 示例
│   ├── chat_arena_eval.py      # ChatArena 评测
│   └── chat_arena_eval_demo.py # Day 18-19 示例
│
├── orchestration/               # 编排模块
│   ├── __init__.py
│   ├── minimal_orchestrator.py # Day 20-21 自研编排器
│   └── pipeline.py             # Pipeline 实现
│
├── app/                         # 应用入口
│   ├── __init__.py
│   └── integrated_rag_system.py # 完整系统集成
│
├── data/                        # 数据目录
│   ├── documents/              # 测试文档
│   └── evaluations/            # 评测数据集
│
├── notebooks/                   # Jupyter  notebooks
│   ├── 01_embedding_exploration.ipynb
│   ├── 02_vector_db_exploration.ipynb
│   └── 03_rag_analysis.ipynb
│
└── docs/                        # 文档目录
    ├── architecture.md         # 系统架构文档
    ├── learning_notes.md      # 学习笔记模板
    └── references.md          # 参考资料汇总
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
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

# 编辑 .env 文件，填入你的 API Keys
# OPENAI_API_KEY=your_key_here
# QDRANT_URL=http://localhost:6333
```

### 3. 启动 Qdrant

```bash
# 使用 Docker 启动 Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
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

---

## ✅ 学习检查清单

### 第 1 周检查清单

- [ ] **Day 1-2**: 能够使用 bge-large-zh 生成句向量并计算相似度
- [ ] **Day 3-4**: 能够使用 Qdrant 存储和检索向量
- [ ] **Day 5-7**: 能够构建一个完整的 RAG 系统（检索 + 重排 + 生成）

### 第 2 周检查清单

- [ ] **Day 8-10**: 能够实现 RAG-Fusion 多查询融合
- [ ] **Day 11-13**: 能够使用 DSPy/Guidance 进行结构化输出
- [ ] **Day 14**: 完成系统整合，能够运行端到端流程

### 第 3 周检查清单

- [ ] **Day 15-17**: 能够使用 Ragas 进行自动化评测
- [ ] **Day 18-19**: 能够使用 ChatArena 进行对话评测
- [ ] **Day 20-21**: 能够设计并实现一个最小化的自研编排器

### 最终成果检查

- [ ] 完成一个完整的 RAG 系统（包含所有核心组件）
- [ ] 能够独立解释每个组件的原理和作用
- [ ] 能够根据评测结果优化系统性能
- [ ] 具备自研编排框架的基础能力

---

## 💡 学习建议

### 1. 理论与实践结合
- 每学习一个概念，立即动手实践
- 不要只看文档，要运行代码、观察结果
- 遇到问题先思考，再查阅资料

### 2. 循序渐进
- 按照 Day 1 → Day 21 的顺序学习
- 每个阶段完成后再进入下一阶段
- 不要跳过基础，直接学习高级内容

### 3. 记录与总结
- 每天记录学习笔记（使用 `docs/learning_notes.md` 模板）
- 记录遇到的问题和解决方案
- 定期回顾和总结

### 4. 实验与对比
- 每个技术都要做对比实验（有 vs 无）
- 记录实验结果，分析原因
- 形成自己的最佳实践

### 5. 社区参与
- 遇到问题查阅 GitHub Issues
- 参与开源社区讨论
- 分享自己的学习心得

---

## 📚 核心参考资料汇总

### 论文
1. [RAG 原始论文](https://arxiv.org/abs/2005.11401) - Retrieval-Augmented Generation
2. [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Sentence Embeddings
3. [BGE 模型](https://arxiv.org/abs/2309.07597) - BAAI General Embedding
4. [RAG-Fusion](https://arxiv.org/abs/2401.10415) - Query Ensemble
5. [DSPy](https://arxiv.org/abs/2310.03714) - Declarative Language Model Calls
6. [Ragas](https://arxiv.org/abs/2309.15217) - Automated Evaluation
7. [HNSW](https://arxiv.org/abs/1603.09320) - Approximate Nearest Neighbor Search

### 文档
1. [LangChain 文档](https://python.langchain.com/)
2. [LlamaIndex 文档](https://docs.llamaindex.ai/)
3. [Qdrant 文档](https://qdrant.tech/documentation/)
4. [Sentence Transformers](https://www.sbert.net/)
5. [DSPy 文档](https://dspy-docs.vercel.app/)
6. [Ragas 文档](https://docs.ragas.io/)

### GitHub 仓库
1. [LangChain](https://github.com/langchain-ai/langchain)
2. [LlamaIndex](https://github.com/run-llama/llama_index)
3. [Qdrant](https://github.com/qdrant/qdrant)
4. [DSPy](https://github.com/stanfordnlp/dspy)
5. [Guidance](https://github.com/microsoft/guidance)
6. [Ragas](https://github.com/explodinggradients/ragas)

---

## 🎓 学习成果

完成本路线后，你将：

1. ✅ **深入理解 RAG 技术栈**：从嵌入到检索到生成的完整流程
2. ✅ **掌握核心技术**：向量数据库、重排、多查询融合、结构化输出
3. ✅ **具备评测能力**：能够定量评估系统性能并优化
4. ✅ **理解编排框架**：掌握 LangChain/LlamaIndex 的设计思想
5. ✅ **具备自研能力**：能够设计并实现自己的编排框架

---

## 📞 获取帮助

- 遇到问题？查看 [GitHub Issues](https://github.com/your-repo/issues)
- 学习笔记？查看 `docs/learning_notes.md`
- 参考资料？查看 `docs/references.md`

---

**祝你学习顺利！🚀**

