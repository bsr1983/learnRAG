"""
Day 3-4: Qdrant 向量数据库演示
演示如何使用 Qdrant 存储和检索向量
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 使用绝对导入避免循环导入问题
from storage.qdrant_wrapper import QdrantClient
from embeddings.embed_model import EmbeddingModel


def check_qdrant_connection():
    """检查 Qdrant 连接"""
    try:
        import requests
        response = requests.get("http://localhost:6333/healthz", timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False


def main():
    print("=" * 60)
    print("Day 3-4: Qdrant 向量数据库演示")
    print("=" * 60)
    print()
    
    # 检查 Qdrant 是否运行
    print("检查 Qdrant 连接...")
    if not check_qdrant_connection():
        print("❌ 错误: 无法连接到 Qdrant")
        print()
        print("请先启动 Qdrant 服务:")
        print("  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print()
        print("或者使用提供的脚本:")
        print("  ./scripts/setup_qdrant.sh")
        print()
        return
    
    print("✅ Qdrant 连接正常")
    print()
    
    # 初始化嵌入模型
    print("正在加载嵌入模型...")
    try:
        embedder = EmbeddingModel(model_name="BAAI/bge-large-zh")
    except Exception as e:
        print(f"⚠️  加载 BAAI/bge-large-zh 失败: {e}")
        print("尝试使用备选模型...")
        embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    
    print(f"✅ 模型加载完成，向量维度: {embedder.get_dimension()}")
    print()
    
    # 初始化 Qdrant 客户端
    print("连接 Qdrant 向量数据库...")
    vector_db = QdrantClient(
        url="http://localhost:6333",
        collection_name="rag_demo"
    )
    print()
    
    # 创建集合
    print("创建集合...")
    vector_size = embedder.get_dimension()
    vector_db.create_collection(vector_size=vector_size)
    print()
    
    # 准备文档
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统",
        "机器学习是人工智能的核心技术，通过算法让计算机从数据中学习模式",
        "深度学习使用多层神经网络来模拟人脑的学习过程",
        "自然语言处理是AI的重要应用领域，使计算机能够理解和生成人类语言",
        "计算机视觉让机器能够理解和分析图像和视频内容",
        "强化学习通过与环境交互来学习最优策略",
        "知识图谱将信息组织成结构化的知识网络",
        "推荐系统使用机器学习算法为用户推荐相关内容"
    ]
    
    print(f"准备 {len(documents)} 个文档")
    print("-" * 60)
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:50]}...")
    print()
    
    # 生成向量
    print("正在生成文档向量...")
    embeddings = embedder.encode(documents, show_progress_bar=True)
    print(f"✅ 成功生成 {len(embeddings)} 个向量")
    print()
    
    # 添加元数据
    metadatas = [
        {"category": "基础概念", "source": "AI基础"},
        {"category": "核心技术", "source": "ML基础"},
        {"category": "核心技术", "source": "DL基础"},
        {"category": "应用领域", "source": "NLP基础"},
        {"category": "应用领域", "source": "CV基础"},
        {"category": "核心技术", "source": "RL基础"},
        {"category": "知识表示", "source": "KG基础"},
        {"category": "应用系统", "source": "推荐系统"}
    ]
    
    # 添加到向量库
    print("正在添加文档到向量库...")
    doc_ids = vector_db.add_documents(
        texts=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"✅ 成功添加 {len(doc_ids)} 个文档")
    print()
    
    # 执行搜索查询
    print("=" * 60)
    print("向量检索演示")
    print("=" * 60)
    print()
    
    queries = [
        "什么是机器学习？",
        "如何让计算机理解图像？",
        "神经网络是如何工作的？"
    ]
    
    for query in queries:
        print(f"查询: {query}")
        print("-" * 60)
        
        # 生成查询向量
        query_vector = embedder.encode(query)
        
        # 搜索
        results = vector_db.search(query_vector, top_k=3)
        
        print(f"找到 {len(results)} 个相关文档:\n")
        for i, result in enumerate(results, 1):
            print(f"  [{i}] 相似度: {result['score']:.4f}")
            print(f"      文档: {result['text']}")
            if result['metadata']:
                print(f"      元数据: {result['metadata']}")
            print()
        print()
    
    # 演示带过滤条件的搜索
    print("=" * 60)
    print("带过滤条件的搜索演示")
    print("=" * 60)
    print()
    
    query = "AI的核心技术是什么？"
    print(f"查询: {query}")
    print("过滤条件: category = '核心技术'")
    print("-" * 60)
    
    query_vector = embedder.encode(query)
    
    # 注意：这里简化处理，实际应该使用 Qdrant 的 Filter
    results = vector_db.search(query_vector, top_k=5)
    
    # 手动过滤
    filtered_results = [
        r for r in results 
        if r.get('metadata', {}).get('category') == '核心技术'
    ]
    
    print(f"找到 {len(filtered_results)} 个匹配的文档:\n")
    for i, result in enumerate(filtered_results, 1):
        print(f"  [{i}] 相似度: {result['score']:.4f}")
        print(f"      文档: {result['text']}")
        print(f"      分类: {result['metadata'].get('category', 'N/A')}")
        print()
    
    print("=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print()
    print("💡 学习要点:")
    print("  1. 向量数据库用于存储和检索高维向量")
    print("  2. 相似度搜索可以快速找到语义相似的文档")
    print("  3. 元数据可以用于过滤和分类")
    print("  4. Qdrant 支持多种距离度量方式（余弦、欧氏距离等）")
    print()
    print("📝 清理数据（可选）:")
    print("  如果需要删除集合，可以运行:")
    print("    vector_db.delete_collection()")


if __name__ == "__main__":
    main()

