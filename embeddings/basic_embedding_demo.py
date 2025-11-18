"""
Day 1-2: 基础嵌入模型演示
演示如何使用 BGE 和 M3E 模型生成文本向量
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embed_model import EmbeddingModel
import numpy as np


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    print("=" * 60)
    print("Day 1-2: 语义嵌入与向量基础示例")
    print("=" * 60)
    print()
    
    # 初始化模型（使用中文模型）
    print("正在加载嵌入模型...")
    print("提示: 首次运行需要从 Hugging Face 下载模型，请确保网络连接正常")
    print("      如果网络不稳定，可以使用镜像源或预先下载模型")
    print()
    
    try:
        # 尝试使用 bge-large-zh 模型
        embedder = EmbeddingModel(model_name="BAAI/bge-large-zh")
    except Exception as e:
        print(f"⚠️  加载 BAAI/bge-large-zh 失败: {e}")
        print("尝试使用更小的模型: all-MiniLM-L6-v2...")
        try:
            # 使用更小的英文模型作为备选
            embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
            print("✅ 使用备选模型: all-MiniLM-L6-v2")
        except Exception as e2:
            print(f"❌ 模型加载失败: {e2}")
            print("\n解决方案:")
            print("1. 检查网络连接")
            print("2. 配置 Hugging Face 镜像源:")
            print("   export HF_ENDPOINT=https://hf-mirror.com")
            print("3. 或使用本地已下载的模型路径")
            return
    
    print(f"✅ 模型加载完成，向量维度: {embedder.get_dimension()}")
    print()
    
    # 测试文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个子领域",
        "今天天气真好，适合出去散步",
        "我喜欢吃苹果和香蕉"
    ]
    
    print("测试文本:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # 生成向量
    print("正在生成向量...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    print(f"\n✅ 成功生成 {len(embeddings)} 个向量")
    print(f"   每个向量维度: {len(embeddings[0])}")
    print()
    
    # 计算相似度矩阵
    print("=" * 60)
    print("文本相似度分析（余弦相似度）")
    print("=" * 60)
    print()
    
    # 计算相似度矩阵
    similarity_matrix = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    
    # 显示相似度矩阵
    print("相似度矩阵:")
    print("-" * 60)
    print(f"{'文本':<15}", end="")
    for i in range(len(texts)):
        print(f"{i+1:>8}", end="")
    print()
    print("-" * 60)
    
    for i, text in enumerate(texts):
        print(f"{i+1:<2} {text[:12]:<12}", end="")
        for j in range(len(texts)):
            print(f"{similarity_matrix[i][j]:>8.3f}", end="")
        print()
    print()
    
    # 找出最相似的文本对
    print("最相似的文本对:")
    print("-" * 60)
    max_sim = -1
    max_pair = None
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = similarity_matrix[i][j]
            if sim > max_sim:
                max_sim = sim
                max_pair = (i, j)
            print(f"文本 {i+1} vs 文本 {j+1}: {sim:.4f}")
            print(f"  '{texts[i]}'")
            print(f"  '{texts[j]}'")
            print()
    
    print("=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    print()
    print("💡 学习要点:")
    print("  1. 嵌入模型将文本转换为固定维度的向量")
    print("  2. 语义相似的文本，其向量在空间中更接近")
    print("  3. 余弦相似度可以衡量两个向量的相似程度")
    print("  4. 归一化后的向量，余弦相似度等于点积")


if __name__ == "__main__":
    main()

