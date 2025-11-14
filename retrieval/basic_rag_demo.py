"""
Basic RAG implementation.
Day 5-7: 构建最小 RAG Demo
"""

from typing import List, Dict
import os
from dotenv import load_dotenv

from embeddings.embed_model import EmbeddingModel
from embeddings.reranker import Reranker
from storage.qdrant_client import QdrantClient

# 加载环境变量
load_dotenv()


class BasicRAG:
    """基础 RAG 系统：检索 + 重排 + 生成"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-zh",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_documents"
    ):
        """初始化 RAG 系统"""
        self.embedder = EmbeddingModel(model_name=embedding_model_name)
        self.reranker = Reranker(model_name=reranker_model_name)
        self.vector_db = QdrantClient(
            url=qdrant_url,
            collection_name=collection_name
        )
        
        # 创建集合
        vector_size = self.embedder.get_dimension()
        self.vector_db.create_collection(vector_size=vector_size)
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档到知识库"""
        embeddings = self.embedder.encode(documents)
        self.vector_db.add_documents(documents, embeddings, metadatas)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
        rerank_top_k: int = 3
    ) -> List[Dict]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 初始检索数量
            use_reranker: 是否使用重排
            rerank_top_k: 重排后返回数量
            
        Returns:
            检索结果列表
        """
        # 1. 向量检索
        query_vector = self.embedder.encode(query)
        results = self.vector_db.search(query_vector, top_k=top_k)
        
        # 2. 重排序（可选）
        if use_reranker and results:
            documents = [r["text"] for r in results]
            rerank_results = self.reranker.rerank(query, documents, top_k=rerank_top_k)
            
            # 重新组织结果
            reranked_results = []
            for idx, score in rerank_results:
                result = results[idx].copy()
                result["rerank_score"] = score
                reranked_results.append(result)
            
            return reranked_results
        
        return results
    
    def generate_answer(
        self,
        query: str,
        context: str,
        llm_provider: str = "openai"
    ) -> str:
        """
        基于检索到的上下文生成答案
        
        Args:
            query: 问题
            context: 检索到的上下文
            llm_provider: LLM 提供商
            
        Returns:
            生成的答案
        """
        # 这里使用简单的 prompt，实际可以使用 LangChain
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说"根据提供的信息，我无法回答这个问题。"

上下文：
{context}

问题：{query}

答案："""
        
        if llm_provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"[LLM Error: {e}] 请配置 OPENAI_API_KEY"
        else:
            return "[请配置 LLM]"
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True
    ) -> Dict:
        """
        完整的 RAG 查询流程
        
        Returns:
            包含检索结果和生成答案的字典
        """
        # 1. 检索
        retrieved_docs = self.retrieve(query, top_k=top_k, use_reranker=use_reranker)
        
        # 2. 构建上下文
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        
        # 3. 生成答案
        answer = self.generate_answer(query, context)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": answer
        }


if __name__ == "__main__":
    # Day 5-7 示例：基础 RAG 系统
    print("=" * 50)
    print("Day 5-7: 基础 RAG Demo")
    print("=" * 50)
    
    # 初始化 RAG 系统
    rag = BasicRAG()
    
    # 准备文档
    documents = [
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习，而无需明确编程。",
        "深度学习是机器学习的一个分支，使用人工神经网络来模拟人脑的学习过程。",
        "自然语言处理（NLP）是人工智能的一个领域，专注于让计算机理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个分支，致力于让机器能够识别和理解图像和视频中的内容。",
        "强化学习是一种机器学习方法，通过与环境交互来学习最优策略。",
        "神经网络是由相互连接的节点（神经元）组成的计算模型，灵感来自生物神经网络。",
        "Transformer 架构是自然语言处理中的一种重要模型架构，被用于 BERT、GPT 等模型。"
    ]
    
    # 添加文档
    print("\n添加文档到知识库...")
    rag.add_documents(documents)
    
    # 测试查询
    queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理的应用有哪些？"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"查询: {query}")
        print('='*50)
        
        result = rag.query(query, top_k=3, use_reranker=True)
        
        print("\n检索到的文档:")
        for i, doc in enumerate(result["retrieved_documents"], 1):
            score = doc.get("rerank_score", doc.get("score", 0))
            print(f"\n  {i}. [分数: {score:.4f}]")
            print(f"     {doc['text'][:100]}...")
        
        print(f"\n生成的答案:")
        print(f"  {result['answer']}")

