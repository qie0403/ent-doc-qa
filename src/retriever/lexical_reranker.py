"""
词法重排器（无额外依赖的重排基线）
Lexical Reranker - dependency-free rerank baseline

为什么需要它：
    项目默认关闭 Rerank（`ENABLE_RERANK=false`），因为 BGE 需要 FlagEmbedding 与
    一个本地模型；一旦开关打开而依赖缺失，`BGEReranker` 会走 `_fallback_rerank`，
    原实现是"保持原顺序 + 伪分数"，等于什么都没做，却让调用方以为已经重排过。

    本模块给降级路径一个**真的能改变排序**的实现：用查询词在文档中的覆盖度、
    词频饱和与位置加权算一个可解释的分数。它不是神经交叉编码器，效果弱于
    bge-reranker，但胜在零依赖、确定性、可复现，也适合作为 A/B 对照的基线。

口径（务必与数字一起说）：
    - 这是**词法**重排，不是语义交叉编码器；它只调整顺序，不改变召回集合；
    - 因此它能让 Recall@1 / MRR 变动，但不会让 Recall@N（N ≥ 候选数）变动。
"""

import math
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document
from loguru import logger

from src.retriever.reranker import RerankResult, Reranker

# 用 jieba 分词；缺失时退化为 2-gram，保证不因分词器问题整体失效
try:
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover - 环境问题
    _HAS_JIEBA = False

# 中文停用词（只保留高频无信息功能词，避免过度过滤）
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "就", "也", "都",
    "而", "并", "但", "被", "把", "对", "从", "到", "为", "以", "这", "那",
    "一个", "什么", "怎么", "如何", "为什么", "哪些", "多少",
}


def _tokenize(text: str) -> List[str]:
    """分词并去停用词"""
    if not text:
        return []
    if _HAS_JIEBA:
        raw = [t.strip() for t in jieba.lcut(text)]
    else:
        # 退化策略：按非中文/字母数字切，再取 2-gram
        chunks = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", text)
        raw = []
        for c in chunks:
            raw.extend([c[i : i + 2] for i in range(max(1, len(c) - 1))])
    return [t for t in raw if t and t not in _STOPWORDS and len(t) > 1]


class LexicalReranker(Reranker):
    """词法重排器：覆盖度 + 词频饱和 + 位置加权"""

    def __init__(self, k1: float = 1.5, length_penalty: float = 0.75):
        """
        Args:
            k1: 词频饱和参数（与 BM25 同义，控制单次重复的增益上限）
            length_penalty: 长度惩罚（0 表示不惩罚长文档）
        """
        self.k1 = k1
        self.length_penalty = length_penalty

    def score(self, query: str, doc_text: str) -> float:
        """计算单篇文档与查询的词法相关度"""
        q_tokens = _tokenize(query)
        if not q_tokens or not doc_text:
            return 0.0

        d_tokens = _tokenize(doc_text)
        if not d_tokens:
            return 0.0

        doc_len = len(d_tokens)
        avg_len = 50.0  # 固定参考长度，保证单文档打分的绝对值稳定、可跨查询比较
        norm = 1.0 - self.length_penalty + self.length_penalty * (doc_len / avg_len)

        total = 0.0
        for token in q_tokens:
            tf = d_tokens.count(token)
            if tf == 0:
                continue
            # 词频饱和：tf 增益收敛，避免长文档靠重复堆分
            sat = (tf * (self.k1 + 1)) / (tf + self.k1)
            # 前缀加权：命中出现在开头（通常是定义/结论句）给更高权重
            head = doc_text.find(token)
            position_boost = 1.15 if 0 <= head < max(1, len(doc_text) // 3) else 1.0
            total += sat * position_boost

        return total / (norm if norm > 0 else 1.0)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """按词法分数重排并截断到 top_n"""
        if not documents:
            return []

        scored = [
            (doc, self.score(query, doc.page_content), idx)
            for idx, doc in enumerate(documents, start=1)
        ]
        # 分数相同时保留原始顺序（稳定排序），避免结果随机
        scored.sort(key=lambda x: (-x[1], x[2]))

        results = [
            RerankResult(
                document=doc,
                score=float(score),
                original_rank=orig_rank,
                new_rank=new_rank,
            )
            for new_rank, (doc, score, orig_rank) in enumerate(scored[:top_n], start=1)
        ]
        logger.debug(f"词法重排完成: {len(documents)} -> {len(results)}")
        return results


def make_lexical_reranker(k1: float = 1.5, length_penalty: float = 0.75) -> LexicalReranker:
    """工厂函数，便于按配置构造"""
    return LexicalReranker(k1=k1, length_penalty=length_penalty)
