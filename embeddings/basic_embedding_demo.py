"""
Day 1-2: 基础嵌入模型演示
演示如何使用 BGE 和 M3E 模型生成文本向量
"""

# 导入 Python 标准库
import sys  # sys 模块提供对 Python 解释器相关功能的访问，比如路径管理
import os   # os 模块提供与操作系统交互的功能，比如文件路径操作

# 添加项目根目录到 Python 路径
# 解释：
# - __file__ 是当前文件的路径（basic_embedding_demo.py）
# - os.path.abspath(__file__) 获取当前文件的绝对路径
# - os.path.dirname() 获取路径的目录部分（去掉文件名）
# - 两次 dirname 是因为：当前文件在 embeddings/ 目录下，需要回到项目根目录
# - sys.path.insert(0, ...) 将路径插入到 Python 模块搜索路径的最前面
#   这样 Python 就能找到项目中的其他模块了
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目中的自定义模块和第三方库
from embeddings.embed_model import EmbeddingModel  # 导入嵌入模型类
import numpy as np  # 导入 numpy 库，并给它起个别名 np（这是 Python 社区的惯例）
                    # numpy 是 Python 中用于科学计算的核心库，主要用于处理数组和矩阵运算


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    参数:
        vec1: 第一个向量（可以是列表或 numpy 数组）
        vec2: 第二个向量（可以是列表或 numpy 数组）
    
    返回:
        余弦相似度值（范围通常在 -1 到 1 之间，1 表示完全相同，0 表示正交，-1 表示完全相反）
    
    余弦相似度公式: cos(θ) = (A·B) / (||A|| * ||B||)
    其中：
    - A·B 是两个向量的点积（内积）
    - ||A|| 是向量 A 的模长（欧几里得范数）
    - ||B|| 是向量 B 的模长
    """
    # np.array() 将输入转换为 numpy 数组
    # numpy 数组比 Python 列表更适合做数学运算，性能也更好
    vec1 = np.array(vec1)  # 将 vec1 转换为 numpy 数组
    vec2 = np.array(vec2)  # 将 vec2 转换为 numpy 数组
    
    # np.dot(vec1, vec2) 计算两个向量的点积（内积）
    # 点积公式：对于向量 [a1, a2, ...] 和 [b1, b2, ...]，点积 = a1*b1 + a2*b2 + ...
    # 
    # np.linalg.norm(vec1) 计算向量的欧几里得范数（模长）
    # 范数公式：||v|| = sqrt(v1² + v2² + ... + vn²)
    # 
    # 整个公式计算的是两个向量夹角的余弦值
    # 值越大，表示两个向量越相似（方向越接近）
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    """
    主函数：演示嵌入模型的基本使用
    
    这个函数展示了如何：
    1. 加载嵌入模型
    2. 将文本转换为向量
    3. 计算文本之间的相似度
    """
    # print() 函数用于输出文本到控制台
    # "=" * 60 表示将字符串 "=" 重复 60 次，用于打印分隔线
    # 这是一种常见的格式化输出方式
    print("=" * 60)
    print("Day 1-2: 语义嵌入与向量基础示例")
    print("=" * 60)
    print()  # 打印一个空行，用于美观
    
    # 初始化模型（使用中文模型）
    print("正在加载嵌入模型...")
    print("提示: 首次运行需要从 Hugging Face 下载模型，请确保网络连接正常")
    print("      如果网络不稳定，可以使用镜像源或预先下载模型")
    print()
    
    # try-except 是 Python 的异常处理机制
    # try: 尝试执行可能出错的代码
    # except: 如果出错，执行这里的代码（错误处理）
    # 这样可以防止程序因为错误而崩溃
    try:
        # 尝试使用 bge-large-zh 模型（中文大模型）
        # EmbeddingModel 是一个类，这里创建了它的一个实例（对象）
        # model_name 是传递给类的参数，指定要使用的模型名称
        embedder = EmbeddingModel(model_name="BAAI/bge-large-zh")
    except Exception as e:
        # 如果上面的代码出错，会执行这里
        # Exception 是所有异常的基类，e 是捕获到的异常对象
        # f"..." 是 f-string（格式化字符串），可以在字符串中嵌入变量
        # {e} 会被替换为异常的具体信息
        print(f"⚠️  加载 BAAI/bge-large-zh 失败: {e}")
        print("尝试使用更小的模型: all-MiniLM-L6-v2...")
        try:
            # 嵌套的 try-except：如果第一个模型失败，尝试第二个
            # 使用更小的英文模型作为备选
            embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
            print("✅ 使用备选模型: all-MiniLM-L6-v2")
        except Exception as e2:
            # 如果第二个模型也失败，打印错误信息并退出函数
            print(f"❌ 模型加载失败: {e2}")
            print("\n解决方案:")
            print("1. 检查网络连接")
            print("2. 配置 Hugging Face 镜像源:")
            print("   export HF_ENDPOINT=https://hf-mirror.com")
            print("3. 或使用本地已下载的模型路径")
            return  # return 语句会立即退出函数，不再执行后面的代码
    
    # embedder.get_dimension() 调用对象的方法，获取向量维度
    # 向量维度是指每个文本被转换成多少个数字（比如 768 维、1024 维等）
    print(f"✅ 模型加载完成，向量维度: {embedder.get_dimension()}")
    print()
    
    # 测试文本
    # 这里定义了一个列表（list），用方括号 [] 表示
    # 列表是 Python 中常用的数据结构，可以存储多个元素
    # 每个元素是一个字符串（用引号括起来的文本）
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的核心技术",
        "深度学习是机器学习的一个子领域",
        "今天天气真好，适合出去散步",
        "我喜欢吃苹果和香蕉"
    ]
    
    print("测试文本:")
    # enumerate() 函数用于同时获取列表的索引和值
    # enumerate(texts, 1) 表示从 1 开始计数（默认从 0 开始）
    # i 是索引（1, 2, 3...），text 是对应的文本内容
    # for 循环会遍历列表中的每个元素
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")  # f-string 格式化输出，{i} 和 {text} 会被替换为实际值
    print()
    
    # 生成向量
    print("正在生成向量...")
    # embedder.encode() 调用嵌入模型的方法，将文本列表转换为向量列表
    # texts: 输入的文本列表
    # show_progress_bar=True: 显示进度条，方便查看处理进度
    # 返回值 embeddings 是一个二维数组（列表的列表）：
    #   - 外层列表：每个元素对应一个文本的向量
    #   - 内层列表：每个元素是向量的一个维度值（数字）
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # len() 函数返回列表的长度（元素个数）
    # len(embeddings) 返回有多少个向量（应该等于文本数量）
    # len(embeddings[0]) 返回第一个向量的维度数
    # embeddings[0] 表示访问列表的第一个元素（索引从 0 开始）
    print(f"\n✅ 成功生成 {len(embeddings)} 个向量")
    print(f"   每个向量维度: {len(embeddings[0])}")
    print()
    
    # 计算相似度矩阵
    print("=" * 60)
    print("文本相似度分析（余弦相似度）")
    print("=" * 60)
    print()
    
    # 计算相似度矩阵
    # np.zeros((行数, 列数)) 创建一个全零的二维数组（矩阵）
    # len(texts) 是文本的数量，所以创建一个 len(texts) x len(texts) 的矩阵
    # 这个矩阵用来存储每对文本之间的相似度
    # 例如：similarity_matrix[0][1] 表示文本 0 和文本 1 的相似度
    similarity_matrix = np.zeros((len(texts), len(texts)))
    
    # 嵌套循环：外层循环遍历每个文本 i，内层循环遍历每个文本 j
    # range(len(texts)) 生成从 0 到 len(texts)-1 的整数序列
    # 例如：如果有 5 个文本，range(5) 生成 [0, 1, 2, 3, 4]
    for i in range(len(texts)):
        for j in range(len(texts)):
            # 计算文本 i 和文本 j 的相似度
            # embeddings[i] 是第 i 个文本的向量
            # embeddings[j] 是第 j 个文本的向量
            # cosine_similarity() 计算这两个向量的余弦相似度
            # 结果存储在矩阵的 [i][j] 位置
            similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    
    # 显示相似度矩阵
    print("相似度矩阵:")
    print("-" * 60)
    
    # 打印表头（第一行）
    # f"{'文本':<15}" 是格式化字符串：
    #   - '文本' 是要显示的文本
    #   - :<15 表示左对齐，占 15 个字符宽度
    # end="" 表示打印后不换行（默认 print() 会换行）
    print(f"{'文本':<15}", end="")
    
    # 打印列号（1, 2, 3, ...）
    # f"{i+1:>8}" 表示右对齐，占 8 个字符宽度
    # i+1 是因为索引从 0 开始，但显示时从 1 开始更直观
    for i in range(len(texts)):
        print(f"{i+1:>8}", end="")
    print()  # 打印完表头后换行
    print("-" * 60)
    
    # 打印每一行的数据
    for i, text in enumerate(texts):
        # text[:12] 是字符串切片，取前 12 个字符
        # 如果文本超过 12 个字符，会被截断
        # f"{i+1:<2}" 表示行号左对齐，占 2 个字符
        # f"{text[:12]:<12}" 表示文本左对齐，占 12 个字符
        print(f"{i+1:<2} {text[:12]:<12}", end="")
        
        # 打印这一行所有文本的相似度值
        for j in range(len(texts)):
            # f"{similarity_matrix[i][j]:>8.3f}" 格式化数字：
            #   - :>8 表示右对齐，占 8 个字符宽度
            #   - .3f 表示保留 3 位小数（f 表示浮点数格式）
            print(f"{similarity_matrix[i][j]:>8.3f}", end="")
        print()  # 打印完一行后换行
    print()
    
    # 找出最相似的文本对
    print("最相似的文本对:")
    print("-" * 60)
    
    # 初始化变量，用于跟踪最大相似度值
    max_sim = -1  # 初始值设为 -1（因为相似度最小是 -1，所以任何值都会比它大）
    max_pair = None  # None 是 Python 中的特殊值，表示"空"或"无"
                     # 这里用来存储最相似文本对的索引
    
    # 遍历所有文本对
    # 注意：j 从 i+1 开始，这样可以避免重复比较（比如比较了 1 vs 2，就不需要再比较 2 vs 1）
    # 同时也避免了文本与自己的比较（i vs i）
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # 获取当前文本对的相似度
            sim = similarity_matrix[i][j]
            
            # if 语句：如果当前相似度大于之前记录的最大值
            # 就更新最大值和对应的文本对
            if sim > max_sim:
                max_sim = sim  # 更新最大相似度值
                max_pair = (i, j)  # (i, j) 是元组（tuple），用来存储一对值
                                   # 元组用圆括号 () 表示，类似于列表，但不可修改
            
            # 打印当前文本对的相似度信息
            # {sim:.4f} 表示保留 4 位小数的浮点数
            print(f"文本 {i+1} vs 文本 {j+1}: {sim:.4f}")
            print(f"  '{texts[i]}'")  # 打印第一个文本
            print(f"  '{texts[j]}'")  # 打印第二个文本
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


# 这是 Python 的一个常见模式
# __name__ 是 Python 的一个特殊变量，表示当前模块的名称
# 当直接运行这个文件时，__name__ 的值是 "__main__"
# 当这个文件被其他文件导入时，__name__ 的值是文件名（不含扩展名）
# 
# 这个判断的作用是：
# - 如果直接运行这个文件（python basic_embedding_demo.py），就执行 main() 函数
# - 如果这个文件被其他文件导入，就不会执行 main() 函数
# 这样可以避免在导入时意外执行代码
if __name__ == "__main__":
    main()  # 调用主函数，开始执行程序

