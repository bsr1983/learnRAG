"""
Ragas evaluation for RAG system.
Day 15-17: Ragas 检索评测
"""

from typing import List, Dict, Optional
import os
import sys
import pandas as pd

# 添加项目根目录到 Python 路径，确保可以导入 llm 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 不是必需的

# 导入统一的 LLM 客户端
try:
    from llm.llm_client import get_llm_client
    LLM_CLIENT_AVAILABLE = True
except ImportError as e:
    LLM_CLIENT_AVAILABLE = False
    print(f"警告: 无法导入 llm_client ({e})，将使用默认配置")


class RagasEvaluation:
    """使用 Ragas 进行 RAG 系统评测"""
    
    def __init__(self, llm_provider: Optional[str] = None, model_name: Optional[str] = None):
        """
        初始化 Ragas 评测器
        
        Args:
            llm_provider: LLM 提供商 (doubao, openai, qwen等)，如果为 None 则从环境变量读取
            model_name: 模型名称，如果为 None 则使用提供商默认模型
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy
            )
            from ragas.llms import llm_factory
            
            self.evaluate = evaluate
            self.metrics = {
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_relevancy": answer_relevancy
            }
            
            # 初始化 LLM 客户端
            self.llm = None
            self.embeddings = None
            
            if LLM_CLIENT_AVAILABLE:
                try:
                    # 使用统一的 LLM 客户端
                    llm_client = get_llm_client(provider=llm_provider)
                    
                    # 使用 Langchain 的 ChatOpenAI 包装器（Ragas 支持 Langchain LLM）
                    # 这样可以兼容所有使用 OpenAI 兼容 API 的提供商（豆包、OpenAI、通义千问等）
                    from langchain_openai import ChatOpenAI
                    
                    langchain_llm = ChatOpenAI(
                        model=llm_client.model_name,
                        openai_api_key=llm_client.api_key,
                        openai_api_base=llm_client.base_url,
                        temperature=llm_client.temperature,
                        max_tokens=llm_client.max_tokens
                    )
                    
                    # Ragas 会自动将 Langchain LLM 包装为 BaseRagasLLM
                    self.llm = langchain_llm
                    print(f"✓ 已配置 LLM: {llm_client.provider} - {llm_client.model_name}")
                except Exception as e:
                    print(f"⚠️  LLM 客户端初始化失败: {e}")
                    print("将使用 Ragas 默认配置（需要 OPENAI_API_KEY）")
                    self.llm = None
            
            # 配置本地 embeddings（使用项目中的 bge-large-zh 模型）
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                
                # 使用本地 embedding 模型，不需要 API key
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-large-zh",
                    model_kwargs={"device": "cpu"},  # 可以根据需要改为 "cuda"
                    encode_kwargs={"normalize_embeddings": True}
                )
                print("✓ 已配置本地 Embeddings: BAAI/bge-large-zh")
            except ImportError:
                print("⚠️  langchain-huggingface 未安装，将使用 Ragas 默认 embeddings（需要 API key）")
                self.embeddings = None
            except Exception as e:
                print(f"⚠️  Embeddings 初始化失败: {e}")
                print("将使用 Ragas 默认配置（需要 API key）")
                self.embeddings = None
            
            self.available = True
        except ImportError as e:
            print(f"Ragas import error: {e}")
            print("Ragas not installed or version incompatible. Run: pip install ragas")
            self.available = False
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: List[str] = None
    ) -> pd.DataFrame:
        """
        评测 RAG 系统
        
        Args:
            questions: 问题列表
            contexts: 每个问题对应的上下文列表（每个问题可能有多个上下文）
            answers: 生成的答案列表
            ground_truths: 标准答案列表（可选）
            
        Returns:
            评测结果 DataFrame
        """
        if not self.available:
            return pd.DataFrame()
        
        # 准备数据
        data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        # 将 pandas DataFrame 转换为 HuggingFace Dataset
        # Ragas 会自动将其转换为 EvaluationDataset
        from datasets import Dataset as HFDataset
        
        df = pd.DataFrame(data)
        dataset = HFDataset.from_pandas(df)
        
        # 选择评测指标
        metrics_list = list(self.metrics.values())
        
        # 运行评测
        try:
            # 如果配置了自定义 LLM 或 embeddings，则传递给 evaluate
            evaluate_kwargs = {
                "dataset": dataset,
                "metrics": metrics_list
            }
            if self.llm is not None:
                evaluate_kwargs["llm"] = self.llm
            if self.embeddings is not None:
                evaluate_kwargs["embeddings"] = self.embeddings
            
            result = self.evaluate(**evaluate_kwargs)
            return result
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "OPENAI_API_KEY" in error_msg or "api_key" in error_msg:
                print("\n⚠️  错误: Ragas 需要 API 密钥才能运行评估")
                print("请设置环境变量或在 .env 文件中配置：")
                print("  - 使用豆包: DOUBAO_API_KEY=your_key, LLM_PROVIDER=doubao")
                print("  - 使用 OpenAI: OPENAI_API_KEY=your_key, LLM_PROVIDER=openai")
                print("  - 使用通义千问: DASHSCOPE_API_KEY=your_key, LLM_PROVIDER=qwen")
                print("\n详细配置请参考 docs/llm_setup_guide.md")
            else:
                print(f"\n⚠️  评估过程中出错: {e}")
            raise
    
    def create_sample_dataset(self) -> Dict:
        """创建示例评测数据集"""
        return {
            "questions": [
                "什么是人工智能？",
                "机器学习和深度学习有什么区别？",
                "自然语言处理的应用有哪些？"
            ],
            "contexts": [
                ["人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"],
                ["机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习。深度学习是机器学习的一个分支，使用人工神经网络。"],
                ["自然语言处理（NLP）是人工智能的一个领域，专注于让计算机理解、解释和生成人类语言。应用包括机器翻译、情感分析、聊天机器人等。"]
            ],
            "answers": [
                "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "机器学习是人工智能的子领域，通过算法从数据中学习。深度学习是机器学习的分支，使用神经网络。",
                "自然语言处理的应用包括机器翻译、情感分析、聊天机器人等。"
            ],
            "ground_truths": [
                "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习。深度学习是机器学习的一个分支，使用人工神经网络来模拟人脑的学习过程。",
                "自然语言处理的应用包括机器翻译、情感分析、聊天机器人、文本摘要、问答系统等。"
            ]
        }


if __name__ == "__main__":
    # Day 15-17 示例：Ragas 评测
    print("=" * 50)
    print("Day 15-17: Ragas 评测示例")
    print("=" * 50)
    
    # 从环境变量读取 LLM 提供商，如果没有则使用默认值
    llm_provider = os.getenv("LLM_PROVIDER", "doubao")
    print(f"\n使用 LLM 提供商: {llm_provider}")
    print("提示: 可通过环境变量 LLM_PROVIDER 指定 (doubao, openai, qwen等)")
    
    evaluator = RagasEvaluation(llm_provider=llm_provider)
    
    if not evaluator.available:
        print("\n请先安装 Ragas: pip install ragas")
        print("\n示例数据集结构:")
        sample = evaluator.create_sample_dataset()
        for key, value in sample.items():
            print(f"\n{key}:")
            for i, item in enumerate(value[:2], 1):
                print(f"  {i}. {str(item)[:80]}...")
    else:
        # 创建示例数据集
        sample = evaluator.create_sample_dataset()
        
        # 运行评测
        print("\n运行评测...")
        result = evaluator.evaluate_rag_system(
            questions=sample["questions"],
            contexts=sample["contexts"],
            answers=sample["answers"],
            ground_truths=sample["ground_truths"]
        )
        
        print("\n评测结果:")
        print(result)

