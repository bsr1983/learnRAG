# LLM 使用示例

## 快速开始

### 使用豆包（推荐）

```bash
# 1. 设置环境变量
export DOUBAO_API_KEY=your_doubao_api_key
export LLM_PROVIDER=doubao

# 2. 运行 RAG 演示
python retrieval/basic_rag_demo.py
```

### 使用 OpenAI

```bash
# 1. 设置环境变量
export OPENAI_API_KEY=your_openai_key
export LLM_PROVIDER=openai

# 2. 运行 RAG 演示
python retrieval/basic_rag_demo.py
```

## 在代码中使用

### 方式一：使用默认配置（从环境变量读取）

```python
from retrieval.basic_rag_demo import BasicRAG

# 自动从环境变量读取 LLM_PROVIDER
rag = BasicRAG()
result = rag.query("什么是人工智能？")
```

### 方式二：在代码中指定提供商

```python
from retrieval.basic_rag_demo import BasicRAG

# 指定使用豆包
rag = BasicRAG(llm_provider="doubao")
result = rag.query("什么是人工智能？")
```

### 方式三：直接使用 LLM 客户端

```python
from llm.llm_client import get_llm_client

# 获取豆包客户端
client = get_llm_client(provider="doubao")

# 生成文本
response = client.generate("你好，请介绍一下你自己")
print(response)
```

## 获取豆包 API Key

1. 访问 [火山引擎控制台](https://console.volcengine.com/)
2. 注册/登录账号
3. 开通豆包服务
4. 在控制台获取 API Key
5. 设置环境变量：`export DOUBAO_API_KEY=your_key`

## 支持的模型提供商

| 提供商 | 环境变量 | 说明 |
|--------|---------|------|
| 豆包 | `DOUBAO_API_KEY` | 推荐，中文支持好，价格便宜 |
| OpenAI | `OPENAI_API_KEY` | 质量稳定，但价格较高 |
| 通义千问 | `DASHSCOPE_API_KEY` | 阿里云，中文支持好 |
| 文心一言 | `ERNIE_API_KEY` | 百度，中文支持好 |
| 智谱 GLM | `ZHIPU_API_KEY` | 智谱 AI |

## 常见问题

**Q: 如何切换不同的模型？**

A: 设置环境变量 `LLM_PROVIDER` 或在代码中指定：

```python
rag = BasicRAG(llm_provider="doubao")  # 使用豆包
rag = BasicRAG(llm_provider="openai")  # 使用 OpenAI
```

**Q: 豆包的 API Key 格式是什么？**

A: 豆包的 API Key 通常是一个字符串，格式类似：`sk-xxxxx...`

**Q: 可以同时配置多个 API Key 吗？**

A: 可以，但每次只能使用一个。设置 `LLM_PROVIDER` 来选择使用哪个。

**Q: 如何测试 LLM 连接？**

A: 运行测试脚本：

```python
from llm.llm_client import get_llm_client

client = get_llm_client(provider="doubao")
response = client.generate("你好")
print(response)
```

