"""
BM25 关键词检索模块 - 基于词频的全文检索
BM25 Keyword Retrieval Module - TF-based Full-text Search

功能说明：
    实现 BM25 关键词检索算法，作为向量检索的补充。BM25 擅长精确匹配关键词，
    与向量检索的语义匹配形成互补，共同构成混合检索体系。

技术选型：
    - rank-bm25：成熟的 BM25 实现
    - jieba：中文分词，提升中文检索效果
    - 倒排索引：基于词频的快速检索

优化策略：
    1. 中文分词：使用 jieba 进行中文分词，提升中文检索效果
    2. 停用词过滤：过滤无意义的停用词
    3. 动态索引：支持增量更新索引
    4. 归一化分数：便于与向量检索分数融合
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from loguru import logger
from langchain_core.documents import Document


@dataclass
class BM25Result:
    """BM25 检索结果"""
    document: Document
    score: float
    rank: int


class BM25Retriever:
    """
    BM25 关键词检索器

    功能：
        - 基于 BM25 算法的关键词检索
        - 支持中文分词
        - 支持增量更新索引
        - 返回带分数的检索结果
    """

    def __init__(self, language: str = "chinese"):
        """
        初始化 BM25 检索器

        Args:
            language: 语言，可选 chinese/english
        """
        self.language = language
        self._documents: List[Document] = []
        self._doc_ids: List[str] = []
        self._corpus: List[List[str]] = []  # 分词后的语料
        self._bm25 = None
        self._jieba_initialized = False

        # 停用词
        self._stopwords = self._load_stopwords()

        logger.info(f"BM25 检索器初始化完成: 语言={language}")

    def _load_stopwords(self) -> set:
        """加载停用词"""
        # 基础停用词
        stopwords = {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '或',
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
        }
        return stopwords

    def _init_jieba(self) -> None:
        """初始化 jieba 分词器"""
        if self._jieba_initialized:
            return

        try:
            import jieba
            # 设置日志级别
            jieba.setLogLevel(60)  # 关闭 jieba 日志
            self._jieba_initialized = True
            logger.info("jieba 分词器初始化成功")
        except ImportError:
            logger.warning("jieba 未安装，将使用简单分词")
            self._jieba_initialized = False

    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        if self.language == "chinese":
            return self._tokenize_chinese(text)
        else:
            return self._tokenize_english(text)

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词"""
        self._init_jieba()

        if self._jieba_initialized:
            import jieba
            words = jieba.lcut(text)
        else:
            # 简单分词：按字符和英文单词分割
            words = []
            # 提取英文单词
            english_words = re.findall(r'[a-zA-Z]+', text)
            words.extend(english_words)
            # 提取中文字符（单字）
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            words.extend(chinese_chars)

        # 过滤停用词和空字符串
        words = [w.strip().lower() for w in words if w.strip() and w.strip().lower() not in self._stopwords]
        return words

    def _tokenize_english(self, text: str) -> List[str]:
        """英文分词"""
        # 简单分词：按空格和标点分割
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self._stopwords]
        return words

    def add_documents(self, documents: List[Document], doc_ids: Optional[List[str]] = None) -> None:
        """
        添加文档到索引

        Args:
            documents: Document 列表
            doc_ids: 文档 ID 列表（可选）
        """
        if not documents:
            return

        # 生成 ID
        if doc_ids is None:
            import uuid
            doc_ids = [str(uuid.uuid4()) for _ in documents]

        # 分词
        for doc, doc_id in zip(documents, doc_ids):
            tokens = self._tokenize(doc.page_content)
            self._documents.append(doc)
            self._doc_ids.append(doc_id)
            self._corpus.append(tokens)

        # 重建 BM25 索引
        self._build_index()

        logger.info(f"BM25 索引添加 {len(documents)} 个文档，总计 {len(self._documents)} 个")

    def _build_index(self) -> None:
        """构建 BM25 索引"""
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
            logger.debug("BM25 索引构建完成")
        except ImportError:
            logger.warning("rank-bm25 未安装，使用简单 BM25 实现")
            self._bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[BM25Result]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            BM25Result 列表（按分数降序排列）
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        if self._bm25 is not None:
            # 使用 rank-bm25 库
            scores = self._bm25.get_scores(query_tokens)
        else:
            # 简单 BM25 实现
            scores = self._simple_bm25(query_tokens)

        # 排序
        scored_docs = list(zip(self._documents, scores, self._doc_ids))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 取 top_k
        results = []
        for rank, (doc, score, doc_id) in enumerate(scored_docs[:top_k], start=1):
            if score > 0:  # 只返回有相关性的结果
                results.append(BM25Result(
                    document=doc,
                    score=score,
                    rank=rank,
                ))

        return results

    def _simple_bm25(self, query_tokens: List[str]) -> List[float]:
        """
        简单 BM25 实现（当 rank-bm25 不可用时使用）

        Args:
            query_tokens: 查询词列表

        Returns:
            每个文档的分数列表
        """
        N = len(self._corpus)
        if N == 0:
            return []

        # BM25 参数
        k1 = 1.5
        b = 0.75

        # 计算平均文档长度
        avgdl = sum(len(doc) for doc in self._corpus) / N if N > 0 else 1

        scores = []
        for doc_tokens in self._corpus:
            score = 0.0
            dl = len(doc_tokens)

            # 计算词频
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            # 计算 IDF 和 BM25 分数
            for token in query_tokens:
                if token not in tf:
                    continue

                # 计算包含该词的文档数
                n_q = sum(1 for doc in self._corpus if token in doc)

                # IDF
                idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

                # BM25 分数
                tf_q = tf[token]
                numerator = tf_q * (k1 + 1)
                denominator = tf_q + k1 * (1 - b + b * dl / avgdl)
                score += idf * numerator / denominator

            scores.append(score)

        return scores

    def get_document_count(self) -> int:
        """获取索引中的文档数量"""
        return len(self._documents)

    def clear(self) -> None:
        """清空索引"""
        self._documents = []
        self._doc_ids = []
        self._corpus = []
        self._bm25 = None
        logger.info("BM25 索引已清空")
