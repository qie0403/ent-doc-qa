"""
检索模块
Retrieval Module

包含：
    - BM25Retriever: BM25 关键词检索器
    - Reranker / BGEReranker / CohereReranker: 重排器
    - LexicalReranker: 零依赖的词法重排器（也是 BGE/Cohere 不可用时的降级实现）
    - HybridRetriever: 混合检索器
    - RetrievalResult: 检索结果数据类
    - RetrievalStats: 检索统计信息数据类
"""

from src.retriever.bm25 import BM25Retriever, BM25Result
from src.retriever.reranker import (
    Reranker,
    BGEReranker,
    CohereReranker,
    RerankResult,
    create_reranker,
)
from src.retriever.lexical_reranker import LexicalReranker
from src.retriever.hybrid import HybridRetriever, RetrievalResult, RetrievalStats

__all__ = [
    "BM25Retriever",
    "BM25Result",
    "Reranker",
    "BGEReranker",
    "CohereReranker",
    "LexicalReranker",
    "RerankResult",
    "create_reranker",
    "HybridRetriever",
    "RetrievalResult",
    "RetrievalStats",
]
