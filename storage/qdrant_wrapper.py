"""
Qdrant vector database client wrapper.
Day 3-4: 向量数据库
"""

from typing import List, Dict, Optional, Any
import uuid

# 直接导入，文件名已重命名，不再冲突
from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantClient:
    """Qdrant 向量数据库客户端封装"""
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "rag_documents"
    ):
        """
        初始化 Qdrant 客户端
        
        Args:
            url: Qdrant 服务地址
            api_key: API 密钥（可选）
            collection_name: 集合名称
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        
        self.client = QdrantSDK(
            url=url,
            api_key=api_key
        )
        
        print(f"Connected to Qdrant at {url}")
    
    def create_collection(
        self,
        vector_size: int,
        distance=None
    ) -> bool:
        """
        创建集合
        
        Args:
            vector_size: 向量维度
            distance: 距离度量方式（COSINE, EUCLID, DOT），默认为 COSINE
            
        Returns:
            是否创建成功
        """
        if distance is None:
            distance = Distance.COSINE
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"Collection '{self.collection_name}' created successfully")
            return True
        except Exception as e:
            print(f"Collection may already exist: {e}")
            return False
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        添加文档到向量库
        
        Args:
            texts: 文本列表
            embeddings: 向量列表
            metadatas: 元数据列表（可选）
            
        Returns:
            文档 ID 列表
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        points = []
        ids = []
        
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text,
                    **metadata
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(points)} documents to collection")
        return ids
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回前 k 个结果
            filter_conditions: 过滤条件（可选）
            
        Returns:
            搜索结果列表，每个结果包含 id, score, payload
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_conditions
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            }
            for result in results
        ]
    
    def delete_collection(self) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False


if __name__ == "__main__":
    # Day 3-4 示例：使用 Qdrant 存储和检索向量
    print("=" * 50)
    print("Day 3-4: 向量数据库示例")
    print("=" * 50)
    
    # 注意：需要先启动 Qdrant 服务
    # docker run -p 6333:6333 qdrant/qdrant
    
    from embeddings.embed_model import EmbeddingModel
    
    # 初始化嵌入模型和向量库
    embedder = EmbeddingModel()
    vector_db = QdrantClient()
    
    # 创建集合
    vector_size = embedder.get_dimension()
    vector_db.create_collection(vector_size=vector_size)
    
    # 准备文档
    documents = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习使用神经网络来模拟人脑",
        "自然语言处理是AI的重要应用领域",
        "计算机视觉让机器能够理解图像"
    ]
    
    # 生成向量
    print("\n生成文档向量...")
    embeddings = embedder.encode(documents)
    
    # 添加到向量库
    print("\n添加到向量库...")
    vector_db.add_documents(documents, embeddings)
    
    # 搜索
    query = "什么是机器学习？"
    print(f"\n查询: {query}")
    query_vector = embedder.encode(query)
    
    results = vector_db.search(query_vector, top_k=3)
    
    print("\n搜索结果:")
    print("-" * 50)
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i} (相似度: {result['score']:.4f}):")
        print(f"  {result['text']}")

