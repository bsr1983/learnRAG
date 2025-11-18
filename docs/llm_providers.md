# LLM 提供商配置指南

本项目支持多种 LLM 提供商，包括 OpenAI、豆包、通义千问、文心一言等。

## 支持的提供商

### 1. 豆包 (Doubao) - 推荐 ⭐

豆包是字节跳动的 AI 模型，支持中文，价格实惠。

**配置方法：**

```bash
# 设置环境变量
export DOUBAO_API_KEY=your_doubao_api_key
export LLM_PROVIDER=doubao  # 可选，设置默认提供商
```

**获取 API Key：**
1. 访问 [火山引擎控制台](https://console.volcengine.com/)
2. 开通豆包服务
3. 获取 API Key

**使用示例：**

```python
from llm.llm_client import get_llm_client

# 使用豆包
client = get_llm_client(provider="doubao")
response = client.generate("你好")
print(response)
```

### 2. OpenAI

**配置方法：**

```bash
export OPENAI_API_KEY=your_openai_api_key
export LLM_PROVIDER=openai  # 可选
```

### 3. 通义千问 (Qwen)

**配置方法：**

```bash
export DASHSCOPE_API_KEY=your_dashscope_api_key
export LLM_PROVIDER=qwen  # 可选
```

**获取 API Key：**
- 访问 [阿里云 DashScope](https://dashscope.console.aliyun.com/)

### 4. 文心一言 (ERNIE)

**配置方法：**

```bash
export ERNIE_API_KEY=your_ernie_api_key
export LLM_PROVIDER=ernie  # 可选
```

**获取 API Key：**
- 访问 [百度智能云](https://cloud.baidu.com/)

### 5. 智谱 GLM (Zhipu)

**配置方法：**

```bash
export ZHIPU_API_KEY=your_zhipu_api_key
export LLM_PROVIDER=zhipu  # 可选
```

**获取 API Key：**
- 访问 [智谱 AI 开放平台](https://open.bigmodel.cn/)

## 在代码中使用

### 方式一：使用环境变量（推荐）

```bash
# 设置环境变量
export DOUBAO_API_KEY=your_key
export LLM_PROVIDER=doubao

# 运行脚本
python retrieval/basic_rag_demo.py
```

### 方式二：在代码中指定

```python
from retrieval.basic_rag_demo import BasicRAG

# 创建 RAG 实例时指定提供商
rag = BasicRAG(llm_provider="doubao")

# 查询
result = rag.query("什么是人工智能？")
```

### 方式三：使用 .env 文件

创建 `.env` 文件：

```env
DOUBAO_API_KEY=your_doubao_api_key
LLM_PROVIDER=doubao
```

代码会自动加载 `.env` 文件中的配置。

## 模型列表

### 豆包模型
- `doubao-pro-4k` - 默认，适合大多数场景
- `doubao-lite-4k` - 轻量版，速度更快
- `doubao-pro-32k` - 长文本版本

### OpenAI 模型
- `gpt-3.5-turbo` - 默认
- `gpt-4` - 更强但更贵
- `gpt-4-turbo` - 最新版本

### 通义千问模型
- `qwen-turbo` - 默认
- `qwen-plus` - 增强版
- `qwen-max` - 最强版本

## 常见问题

### Q: 如何切换不同的模型提供商？

A: 有两种方式：
1. 设置环境变量 `LLM_PROVIDER=doubao`
2. 在代码中指定：`BasicRAG(llm_provider="doubao")`

### Q: 如何在同一项目中同时使用多个提供商？

A: 可以在不同地方使用不同的提供商：

```python
# 使用豆包生成答案
rag1 = BasicRAG(llm_provider="doubao")

# 使用 OpenAI 生成答案
rag2 = BasicRAG(llm_provider="openai")
```

### Q: 豆包的 API Key 在哪里获取？

A: 
1. 访问 https://console.volcengine.com/
2. 注册/登录账号
3. 开通豆包服务
4. 在控制台获取 API Key

### Q: 支持本地模型吗？

A: 目前主要支持云端 API。如果需要本地模型，可以：
1. 使用兼容 OpenAI API 格式的本地服务（如 LocalAI）
2. 修改 `llm/llm_client.py` 添加本地模型支持

## 价格对比（仅供参考）

- **豆包**: 最便宜，适合中文场景 ⭐推荐
- **通义千问**: 价格适中，中文支持好
- **OpenAI**: 价格较高，但质量稳定
- **文心一言**: 价格适中
- **智谱 GLM**: 价格适中

## 更新日志

- 2024-11: 添加豆包、通义千问、文心一言、智谱 GLM 支持
- 2024-10: 初始版本，仅支持 OpenAI

