"""
Embedding model wrapper for bge-large-zh and m3e-large.
Day 1-2: 语义嵌入与向量基础
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    """封装嵌入模型，支持 bge-large-zh 和 m3e-large"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh",
        device: str = None,
        normalize_embeddings: bool = True
    ):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称，支持 "BAAI/bge-large-zh" 或 "moka-ai/m3e-large"
            device: 设备，"cuda" 或 "cpu"，None 表示自动选择
            normalize_embeddings: 是否归一化向量
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings = normalize_embeddings
        
        print(f"Loading embedding model: {model_name}")
        print(f"Device: {self.device}")
        
        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
            
        Returns:
            单个向量或向量列表
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings
        )
        
        if len(embeddings) == 1:
            return embeddings[0].tolist()
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Day 1-2 示例：使用 bge-large-zh 生成句向量
    print("=" * 50)
    print("Day 1-2: 语义嵌入与向量基础示例")
    print("=" * 50)
    
    # 初始化模型
    embedder = EmbeddingModel(model_name="BAAI/bge-large-zh")
    
    # 测试文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个子领域",
        "今天天气真好，适合出去散步",
        "我喜欢吃苹果和香蕉"
    ]
    
    # 生成向量
    print("\n生成向量...")
    embeddings = embedder.encode(texts)
    
    print(f"\n向量维度: {embedder.get_dimension()}")
    print(f"生成了 {len(embeddings)} 个向量")
    
    # 计算相似度（使用余弦相似度）
    import numpy as np
    
    def cosine_similarity(vec1, vec2):
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    print("\n文本相似度矩阵:")
    print("-" * 50)
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts[i+1:], start=i+1):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"文本 {i+1} vs 文本 {j+1}: {sim:.4f}")
            print(f"  '{text1[:30]}...'")
            print(f"  '{text2[:30]}...'")
            print()

