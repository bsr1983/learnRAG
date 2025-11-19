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

# 确保从项目根目录加载 .env 文件
load_dotenv(dotenv_path=os.path.join(project_root, '.env'))


class IntegratedRAGSystem:
    """完整的 RAG 系统：整合所有组件"""
    
    def __init__(
        self,
        use_rag_fusion: bool = True,
        use_reranker: bool = True,
        use_structured_output: bool = False,
        llm_provider: str = None  # None 表示从环境变量读取
    ):
        """
        初始化完整 RAG 系统
        
        Args:
            use_rag_fusion: 是否使用 RAG-Fusion
            use_reranker: 是否使用重排
            use_structured_output: 是否使用结构化输出
            llm_provider: LLM 提供商（doubao, openai, qwen 等）
        """
        self.use_rag_fusion = use_rag_fusion
        self.use_reranker = use_reranker
        self.use_structured_output = use_structured_output
        
        # 初始化组件（传递 llm_provider）
        self.basic_rag = BasicRAG(llm_provider=llm_provider)
        if use_rag_fusion:
            self.rag_fusion = RAGFusion(llm_provider=llm_provider)
        if use_structured_output:
            self.structured_output = StructuredOutputDemo()
        
        # 标记是否已添加文档（避免重复添加）
        self._documents_added = False
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, force: bool = False):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表
            force: 是否强制添加（即使已添加过）
        """
        if self._documents_added and not force:
            print("⚠️  文档已添加过，跳过。如需重新添加，请设置 force=True 或清理集合")
            return
        
        self.basic_rag.add_documents(documents, metadatas)
        self._documents_added = True
    
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
                top_k_per_query=8,  # 增加每个查询的检索数量
                final_top_k=8  # 增加最终返回数量
            )
        else:
            retrieved_docs = self.basic_rag.retrieve(
                query,
                top_k=8,  # 增加检索数量
                use_reranker=self.use_reranker
            )
        
        # 2. 构建上下文（去重并格式化）
        seen_texts = set()
        unique_docs = []
        seen_ids = set()
        
        for doc in retrieved_docs:
            # 优先使用 ID 去重，如果没有 ID 则使用文本内容
            doc_id = doc.get("id")
            text = doc.get("text", "").strip()
            
            # 去重逻辑：优先使用 ID，否则使用文本内容
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
            elif text and text not in seen_texts:
                seen_texts.add(text)
                unique_docs.append(doc)
        
        # 如果去重后没有文档，使用原始结果的前几个
        if not unique_docs:
            print("⚠️  警告: 去重后没有文档，使用原始结果")
            unique_docs = retrieved_docs[:min(5, len(retrieved_docs))]
        
        # 格式化上下文，添加更多信息
        context_parts = []
        for i, doc in enumerate(unique_docs, 1):
            text = doc.get("text", "").strip()
            if not text:
                continue
            score = doc.get("fusion_score") or doc.get("rerank_score") or doc.get("score", 0)
            context_parts.append(f"文档{i}（相关度: {score:.3f}）: {text}")
        
        context = "\n\n".join(context_parts)
        
        # 调试信息
        print(f"\n📝 上下文构建:")
        print(f"   检索总数: {len(retrieved_docs)}")
        print(f"   去重后: {len(unique_docs)} 个唯一文档")
        print(f"   上下文长度: {len(context)} 字符")
        
        # 如果上下文为空或太短，给出提示
        if not context or len(context) < 50:
            print("⚠️  警告: 检索到的上下文内容较少，可能影响答案质量")
            print(f"   上下文预览: {context[:200]}...")
        
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
    
    # 检查环境变量
    llm_provider = os.getenv("LLM_PROVIDER")
    doubao_key = os.getenv("DOUBAO_API_KEY")
    
    if not doubao_key and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  警告: 未检测到 LLM API Key")
        print("请设置环境变量:")
        print("  export DOUBAO_API_KEY=your_key")
        print("  export LLM_PROVIDER=doubao")
        print("或者创建 .env 文件")
        print()
    
    # 初始化系统（传递 llm_provider）
    system = IntegratedRAGSystem(
        use_rag_fusion=True,
        use_reranker=True,
        use_structured_output=True,
        llm_provider=llm_provider  # 从环境变量读取
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
    # 统计唯一文档
    unique_texts = set()
    for doc in result["retrieved_documents"]:
        unique_texts.add(doc.get("text", "").strip())
    
    print(f"检索总数: {len(result['retrieved_documents'])}")
    print(f"唯一文档: {len(unique_texts)}")
    print()
    
    # 显示去重后的文档
    seen_texts = set()
    for i, doc in enumerate(result["retrieved_documents"], 1):
        text = doc.get("text", "").strip()
        if text in seen_texts:
            continue  # 跳过重复文档
        seen_texts.add(text)
        
        # 优先显示 fusion_score，然后是 rerank_score，最后是 score
        if "fusion_score" in doc:
            score = doc["fusion_score"]
            score_type = "融合分数"
        elif "rerank_score" in doc:
            score = doc["rerank_score"]
            score_type = "重排分数"
        else:
            score = doc.get("score", 0)
            score_type = "相似度"
        print(f"\n  {len(seen_texts)}. [{score_type}: {score:.4f}]")
        print(f"     {text[:150]}...")
    
    print(f"\n【生成的答案】")
    print(f"  {result['answer']}")

