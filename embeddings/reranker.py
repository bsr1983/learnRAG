"""
Reranker model wrapper for bge-reranker-base.
Day 5-7: 重排模型集成
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
import torch


class Reranker:
    """封装重排模型（cross-encoder）"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = None
    ):
        """
        初始化重排模型
        
        Args:
            model_name: 模型名称
            device: 设备，"cuda" 或 "cpu"
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading reranker model: {model_name}")
        print(f"Device: {self.device}")
        
        self.model = CrossEncoder(
            model_name,
            device=self.device
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果，None 表示返回全部
            
        Returns:
            List of (index, score) tuples, 按分数降序排列
        """
        if not documents:
            return []
        
        # 构建 query-document 对
        pairs = [[query, doc] for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 排序并返回索引和分数
        results = [(i, float(score)) for i, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results


if __name__ == "__main__":
    # Day 5-7 示例：使用重排模型
    print("=" * 50)
    print("Day 5-7: 重排模型示例")
    print("=" * 50)
    
    # 初始化重排模型
    reranker = Reranker()
    
    # 测试查询和文档
    query = "什么是人工智能？"
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习。",
        "今天天气很好，适合出去散步。",
        "深度学习使用神经网络来模拟人脑的学习过程。",
        "我喜欢吃苹果和香蕉。"
    ]
    
    # 重排序
    print(f"\n查询: {query}")
    print(f"\n文档数量: {len(documents)}")
    print("\n重排序结果:")
    print("-" * 50)
    
    results = reranker.rerank(query, documents, top_k=3)
    
    for rank, (idx, score) in enumerate(results, 1):
        print(f"\n排名 {rank} (分数: {score:.4f}):")
        print(f"文档 {idx+1}: {documents[idx][:80]}...")

