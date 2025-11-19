# RAG 系统问题排查

## 常见问题及解决方案

### 1. 检索分数很低（< 0.1）

**问题现象：**
- 检索到的文档相似度分数很低（如 0.04）
- 返回的文档似乎不相关

**可能原因：**
1. 文档内容与查询不匹配
2. 向量维度不匹配
3. 距离度量方式不合适

**解决方案：**
```python
# 检查向量维度
from embeddings.embed_model import EmbeddingModel
embedder = EmbeddingModel()
print(f"向量维度: {embedder.get_dimension()}")

# 检查 Qdrant 集合配置
from storage.qdrant_wrapper import QdrantClient
client = QdrantClient()
# 确保集合的 vector_size 与嵌入模型维度一致
```

### 2. 文档重复

**问题现象：**
- RAG-Fusion 返回的文档有重复
- 多个查询返回相同的文档

**解决方案：**
已修复！RRF 算法现在会正确去重：
- 使用文档 ID 或文本内容作为唯一标识
- 保留相似度分数更高的文档
- 在构建上下文时也会去重

### 3. LLM 返回"根据提供的信息，我无法回答这个问题"

**问题现象：**
- 检索到了相关文档
- 但 LLM 说无法回答

**可能原因：**
1. **未配置 LLM API Key**（最常见）
2. LLM 客户端初始化失败
3. 上下文确实不够相关

**解决方案：**

#### 配置豆包 API Key（推荐）

```bash
# 设置环境变量
export DOUBAO_API_KEY=your_doubao_api_key
export LLM_PROVIDER=doubao

# 或者在代码中指定
python -c "
from app.integrated_rag_system import IntegratedRAGSystem
system = IntegratedRAGSystem(llm_provider='doubao')
"
```

#### 配置 OpenAI API Key

```bash
export OPENAI_API_KEY=your_openai_key
export LLM_PROVIDER=openai
```

#### 检查 LLM 客户端状态

```python
from llm.llm_client import get_llm_client

try:
    client = get_llm_client(provider="doubao")
    response = client.generate("测试")
    print(f"✅ LLM 连接成功: {response}")
except Exception as e:
    print(f"❌ LLM 连接失败: {e}")
```

### 4. Tokenizers 警告

**问题现象：**
```
huggingface/tokenizers: The current process just got forked...
```

**解决方案：**
```bash
# 设置环境变量（在运行脚本前）
export TOKENIZERS_PARALLELISM=false

# 或者在代码开头添加
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### 5. 检索结果不相关

**问题现象：**
- 检索到的文档与查询不匹配
- 分数很低但文档内容看起来相关

**解决方案：**

1. **检查嵌入模型是否适合你的数据**
   - 中文数据使用 `BAAI/bge-large-zh`
   - 英文数据可以使用 `all-MiniLM-L6-v2`

2. **使用重排序器**
   ```python
   rag = BasicRAG(use_reranker=True)
   ```

3. **调整检索参数**
   ```python
   # 增加检索数量
   results = rag.retrieve(query, top_k=10, use_reranker=True, rerank_top_k=5)
   ```

### 6. RAG-Fusion 生成的查询质量差

**问题现象：**
- 生成的查询与原始查询太相似
- 或者生成的查询没有意义

**解决方案：**

1. **确保 LLM 客户端正确配置**
   ```bash
   export DOUBAO_API_KEY=your_key
   export LLM_PROVIDER=doubao
   ```

2. **调整查询生成参数**
   ```python
   rag_fusion = RAGFusion(llm_provider="doubao")
   queries = rag_fusion.generate_queries(
       original_query="什么是AI？",
       num_queries=3
   )
   ```

### 7. 集合已存在错误

**问题现象：**
```
Collection may already exist: Unexpected Response: 409 (Conflict)
```

**解决方案：**
这是正常的！集合已存在，可以继续使用。如果需要重新开始：

```python
from storage.qdrant_wrapper import QdrantClient

# 删除现有集合
client = QdrantClient(collection_name="rag_documents")
client.delete_collection()

# 重新创建
client.create_collection(vector_size=1024)
```

## 调试技巧

### 1. 检查检索结果

```python
from retrieval.basic_rag_demo import BasicRAG

rag = BasicRAG()
results = rag.retrieve("你的查询", top_k=5)

for i, doc in enumerate(results, 1):
    print(f"{i}. 分数: {doc['score']:.4f}")
    print(f"   文档: {doc['text'][:100]}...")
    print()
```

### 2. 检查向量相似度

```python
from embeddings.embed_model import EmbeddingModel
import numpy as np

embedder = EmbeddingModel()

query = "什么是人工智能？"
doc = "人工智能是计算机科学的一个分支"

query_vec = embedder.encode(query)
doc_vec = embedder.encode(doc)

# 计算余弦相似度
similarity = np.dot(query_vec, doc_vec) / (
    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
)
print(f"相似度: {similarity:.4f}")
```

### 3. 检查 LLM 响应

```python
from llm.llm_client import get_llm_client

client = get_llm_client(provider="doubao")
response = client.generate("你好")
print(response)
```

## 性能优化建议

1. **使用 GPU**（如果有）
   ```python
   embedder = EmbeddingModel(device="cuda")
   ```

2. **批量处理**
   ```python
   embeddings = embedder.encode(documents, batch_size=32)
   ```

3. **缓存结果**
   - 向量可以预先计算并存储
   - LLM 响应可以缓存（如果查询重复）

## 获取帮助

如果以上方案都无法解决问题：
1. 检查日志输出
2. 查看 `docs/llm_providers.md` 了解 LLM 配置
3. 查看 `docs/setup_troubleshooting.md` 了解环境配置

