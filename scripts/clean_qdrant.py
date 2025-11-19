#!/usr/bin/env python3
"""
清理 Qdrant 集合并重新初始化
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from storage.qdrant_wrapper import QdrantClient

print("=" * 60)
print("清理 Qdrant 集合")
print("=" * 60)
print()

# 删除现有集合
client = QdrantClient(collection_name="rag_documents")
print("正在删除集合 'rag_documents'...")
try:
    client.delete_collection()
    print("✅ 集合已删除")
except Exception as e:
    print(f"⚠️  删除集合时出错（可能不存在）: {e}")

print()
print("✅ 清理完成！")
print("现在可以重新运行 RAG 系统，文档将被重新添加")

