"""
Integrated RAG system with all components.
Day 14: 系统整合
"""

import sys
import os
from typing import Dict, List
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from retrieval.basic_rag_demo import BasicRAG
from retrieval.rag_fusion_demo import RAGFusion
from llm.structured_output_demo import StructuredOutputDemo

load_dotenv()


class IntegratedRAGSystem:
    """完整的 RAG 系统：整合所有组件"""
    
    def __init__(
        self,
        use_rag_fusion: bool = True,
        use_reranker: bool = True,
        use_structured_output: bool = False
    ):
        """
        初始化完整 RAG 系统
        
        Args:
            use_rag_fusion: 是否使用 RAG-Fusion
            use_reranker: 是否使用重排
            use_structured_output: 是否使用结构化输出
        """
        self.use_rag_fusion = use_rag_fusion
        self.use_reranker = use_reranker
        self.use_structured_output = use_structured_output
        
        # 初始化组件
        self.basic_rag = BasicRAG()
        if use_rag_fusion:
            self.rag_fusion = RAGFusion()
        if use_structured_output:
            self.structured_output = StructuredOutputDemo()
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档到知识库"""
        self.basic_rag.add_documents(documents, metadatas)
    
    def query(
        self,
        query: str,
        return_structured: bool = False,
        output_fields: List[str] = None
    ) -> Dict:
        """
        完整的 RAG 查询流程
        
        Args:
            query: 查询文本
            return_structured: 是否返回结构化输出
            output_fields: 结构化输出的字段列表
            
        Returns:
            包含检索结果和生成答案的字典
        """
        # 1. 检索（使用 RAG-Fusion 或基础 RAG）
        if self.use_rag_fusion:
            retrieved_docs = self.rag_fusion.retrieve_fusion(
                query,
                num_queries=3,
                top_k_per_query=5,
                final_top_k=5
            )
        else:
            retrieved_docs = self.basic_rag.retrieve(
                query,
                top_k=5,
                use_reranker=self.use_reranker
            )
        
        # 2. 构建上下文
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        
        # 3. 生成答案
        answer = self.basic_rag.generate_answer(query, context)
        
        # 4. 结构化输出（可选）
        structured_data = None
        if return_structured and self.use_structured_output:
            if output_fields:
                structured_data = self.structured_output.simple_extract(
                    answer,
                    output_fields
                )
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "context": context,
            "answer": answer,
            "structured_output": structured_data
        }


if __name__ == "__main__":
    # Day 14 示例：完整系统整合
    print("=" * 50)
    print("Day 14: 完整 RAG 系统整合")
    print("=" * 50)
    
    # 初始化系统
    system = IntegratedRAGSystem(
        use_rag_fusion=True,
        use_reranker=True,
        use_structured_output=True
    )
    
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
    system.add_documents(documents)
    
    # 测试查询
    query = "请详细介绍人工智能的主要技术分支"
    
    print(f"\n{'='*50}")
    print(f"查询: {query}")
    print('='*50)
    
    result = system.query(query, return_structured=False)
    
    print("\n【检索到的文档】")
    for i, doc in enumerate(result["retrieved_documents"], 1):
        score = doc.get("fusion_score", doc.get("rerank_score", doc.get("score", 0)))
        print(f"\n  {i}. [分数: {score:.4f}]")
        print(f"     {doc['text'][:100]}...")
    
    print(f"\n【生成的答案】")
    print(f"  {result['answer']}")

