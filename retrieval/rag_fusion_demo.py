"""
RAG-Fusion implementation: Multi-query retrieval and fusion.
Day 8-10: RAG-Fusion 与查询增强
"""

import sys
import os
from typing import List, Dict
from dotenv import load_dotenv
from collections import defaultdict

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from embeddings.embed_model import EmbeddingModel
from storage.qdrant_wrapper import QdrantClient
from llm.llm_client import get_llm_client

load_dotenv()


class RAGFusion:
    """RAG-Fusion: 多查询融合检索"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-zh",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_documents",
        llm_provider: str = None  # None 表示从环境变量读取
    ):
        """初始化 RAG-Fusion 系统"""
        self.embedder = EmbeddingModel(model_name=embedding_model_name)
        self.vector_db = QdrantClient(
            url=qdrant_url,
            collection_name=collection_name
        )
        # 初始化 LLM 客户端
        try:
            self.llm_client = get_llm_client(provider=llm_provider)
        except Exception as e:
            print(f"警告: LLM 客户端初始化失败: {e}")
            self.llm_client = None
    
    def generate_queries(
        self,
        original_query: str,
        num_queries: int = 3,
        llm_provider: str = None
    ) -> List[str]:
        """
        使用 LLM 生成多个改写查询
        
        Args:
            original_query: 原始查询
            num_queries: 生成查询数量
            llm_provider: LLM 提供商（如果为 None，使用初始化时的设置）
            
        Returns:
            改写后的查询列表
        """
        prompt = f"""基于以下问题，生成 {num_queries} 个不同角度的问题，这些问题应该：
1. 从不同角度询问相同或相关的信息
2. 使用不同的表达方式
3. 涵盖问题的不同方面

原始问题：{original_query}

请只返回问题列表，每行一个问题，不要编号："""
        
        # 如果没有 LLM 客户端，尝试创建
        llm_client = self.llm_client
        if llm_client is None:
            try:
                llm_client = get_llm_client(provider=llm_provider)
            except Exception as e:
                print(f"[LLM Error: {e}] 使用原始查询")
                return [original_query]
        
        try:
            response = llm_client.generate(
                prompt=prompt,
                temperature=0.7
            )
            queries = response.strip().split("\n")
            queries = [q.strip() for q in queries if q.strip()]
            # 确保包含原始查询
            if original_query not in queries:
                queries.insert(0, original_query)
            return queries[:num_queries]
        except Exception as e:
            print(f"[LLM Error: {e}] 使用原始查询")
            # 简单的启发式改写（如果没有 LLM）
            return [
                original_query,
                f"请详细解释{original_query}",
                f"关于{original_query}，你能告诉我什么？"
            ][:num_queries]
    
    def reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) 算法
        
        Args:
            ranked_lists: 多个排序列表
            k: RRF 参数，通常为 60
            
        Returns:
            融合后的排序列表
        """
        # 计算每个文档的 RRF 分数
        doc_scores = defaultdict(float)
        doc_map = {}  # 存储文档内容，使用文本内容作为唯一标识
        
        for list_idx, ranked_list in enumerate(ranked_lists):
            for rank, doc in enumerate(ranked_list, 1):
                # 使用文档 ID 作为唯一标识（优先），如果没有 ID 则使用文本内容
                doc_id = doc.get("id")
                doc_text = doc.get("text", "").strip()
                
                # 如果没有 ID，使用文本内容的 hash 作为 ID
                if not doc_id:
                    import hashlib
                    doc_id = hashlib.md5(doc_text.encode()).hexdigest()
                
                # 如果文档已存在，保留分数更高的那个
                if doc_id in doc_map:
                    # 保留原始相似度分数更高的文档
                    existing_score = doc_map[doc_id].get("score", 0)
                    current_score = doc.get("score", 0)
                    if current_score > existing_score:
                        doc_map[doc_id] = doc.copy()
                else:
                    doc_map[doc_id] = doc.copy()
                
                # 累加 RRF 分数
                doc_scores[doc_id] += 1 / (k + rank)
        
        # 按 RRF 分数排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回融合后的文档列表
        fused_results = []
        for doc_id, fusion_score in sorted_docs:
            doc = doc_map[doc_id].copy()
            doc["fusion_score"] = fusion_score
            # 保留原始相似度分数
            if "score" not in doc:
                doc["score"] = 0.0
            fused_results.append(doc)
        
        return fused_results
    
    def retrieve_fusion(
        self,
        query: str,
        num_queries: int = 3,
        top_k_per_query: int = 5,
        final_top_k: int = 5
    ) -> List[Dict]:
        """
        RAG-Fusion 检索流程
        
        Args:
            query: 原始查询
            num_queries: 生成查询数量
            top_k_per_query: 每个查询检索的数量
            final_top_k: 最终返回数量
            
        Returns:
            融合后的检索结果
        """
        # 1. 生成多个查询
        queries = self.generate_queries(query, num_queries=num_queries)
        print(f"生成的查询 ({len(queries)} 个):")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        
        # 2. 对每个查询进行检索
        all_results = []
        for i, q in enumerate(queries, 1):
            query_vector = self.embedder.encode(q)
            results = self.vector_db.search(query_vector, top_k=top_k_per_query)
            all_results.append(results)
            # 调试信息
            print(f"  查询 {i} 检索到 {len(results)} 个结果")
            if results:
                unique_texts = set(r.get("text", "") for r in results)
                print(f"    唯一文档: {len(unique_texts)}")
        
        # 3. 融合结果（使用 RRF）
        fused_results = self.reciprocal_rank_fusion(all_results)
        
        # 调试信息：检查融合后的结果
        unique_after_fusion = set(doc.get("text", "") for doc in fused_results)
        print(f"\n融合后: {len(fused_results)} 个结果，{len(unique_after_fusion)} 个唯一文档")
        
        return fused_results[:final_top_k]


if __name__ == "__main__":
    # Day 8-10 示例：RAG-Fusion
    print("=" * 50)
    print("Day 8-10: RAG-Fusion 示例")
    print("=" * 50)
    
    from retrieval.basic_rag_demo import BasicRAG
    
    # 初始化系统
    rag = BasicRAG()
    rag_fusion = RAGFusion()
    
    # 准备文档（复用之前的文档）
    documents = [
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习，而无需明确编程。",
        "深度学习是机器学习的一个分支，使用人工神经网络来模拟人脑的学习过程。",
        "自然语言处理（NLP）是人工智能的一个领域，专注于让计算机理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个分支，致力于让机器能够识别和理解图像和视频中的内容。"
    ]
    
    # 添加文档
    print("\n添加文档到知识库...")
    rag.add_documents(documents)
    
    # 测试查询
    query = "人工智能有哪些主要技术？"
    
    print(f"\n{'='*50}")
    print(f"原始查询: {query}")
    print('='*50)
    
    # 基础 RAG（单查询）
    print("\n【基础 RAG（单查询）】")
    basic_results = rag.retrieve(query, top_k=3, use_reranker=False)
    print("检索结果:")
    for i, doc in enumerate(basic_results, 1):
        print(f"  {i}. [分数: {doc['score']:.4f}] {doc['text'][:80]}...")
    
    # RAG-Fusion（多查询融合）
    print("\n【RAG-Fusion（多查询融合）】")
    fusion_results = rag_fusion.retrieve_fusion(
        query,
        num_queries=3,
        top_k_per_query=5,
        final_top_k=3
    )
    print("\n融合后的检索结果:")
    for i, doc in enumerate(fusion_results, 1):
        score = doc.get("fusion_score", doc.get("score", 0))
        print(f"  {i}. [融合分数: {score:.4f}] {doc['text'][:80]}...")

