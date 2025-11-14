"""
Ragas evaluation for RAG system.
Day 15-17: Ragas 检索评测
"""

from typing import List, Dict
import pandas as pd


class RagasEvaluation:
    """使用 Ragas 进行 RAG 系统评测"""
    
    def __init__(self):
        """初始化 Ragas 评测器"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevance
            )
            self.evaluate = evaluate
            self.metrics = {
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_relevance": answer_relevance
            }
            self.available = True
        except ImportError:
            print("Ragas not installed. Run: pip install ragas")
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
        
        dataset = pd.DataFrame(data)
        
        # 选择评测指标
        metrics_list = list(self.metrics.values())
        
        # 运行评测
        result = self.evaluate(
            dataset=dataset,
            metrics=metrics_list
        )
        
        return result
    
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
    
    evaluator = RagasEvaluation()
    
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

