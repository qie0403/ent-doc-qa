"""
混合检索模块 - 向量检索 + BM25 + Rerank
Hybrid Retrieval Module - Vector + BM25 + Rerank

功能说明：
    实现混合检索策略，结合向量检索的语义理解能力和 BM25 的关键词匹配能力，
    再通过 Rerank 精排，最终输出高质量的检索结果。

技术选型：
    - 向量检索：语义相似度匹配，擅长同义、模糊查询
    - BM25 检索：关键词精确匹配，擅长专有名词、术语查询
    - Rerank 精排：交叉编码器计算精确相关性
    - 分数融合：加权融合两种检索方式的分数

优化策略：
    1. 双通道检索：向量 + BM25 并行检索，覆盖更多相关文档
    2. 分数归一化：将不同检索方式的分数映射到统一范围
    3. 加权融合：根据场景调整向量和 BM25 的权重
    4. Rerank 精排：从 top_k 中精选 top_n，大幅提升准确率
    5. 去重合并：合并两个通道的结果，去除重复
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from loguru import logger
from langchain_core.documents import Document

from src.retriever.bm25 import BM25Retriever, BM25Result
from src.retriever.reranker import Reranker, RerankResult


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    rank: int
    source: str = "hybrid"  # vector/bm25/hybrid
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RetrievalStats:
    """检索统计信息"""
    query: str
    total_candidates: int = 0
    vector_count: int = 0
    bm25_count: int = 0
    merged_count: int = 0
    reranked_count: int = 0
    final_count: int = 0
    latency_ms: float = 0.0


class HybridRetriever:
    """
    混合检索器

    检索流程：
        1. 向量检索：获取 top_k 个语义相似结果
        2. BM25 检索：获取 top_k 个关键词匹配结果
        3. 结果合并：去重并加权融合分数
        4. Rerank 精排：使用重排模型重新排序
        5. 返回结果：输出最终的 top_n 个结果
    """

    def __init__(
        self,
        vector_store=None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[Reranker] = None,
        top_k: int = 10,
        rerank_top_n: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        enable_rerank: bool = True,
    ):
        """
        初始化混合检索器

        Args:
            vector_store: 向量存储实例
            bm25_retriever: BM25 检索器实例
            reranker: 重排器实例
            top_k: 初始检索返回数量
            rerank_top_n: 重排后返回数量
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
            enable_rerank: 是否启用重排
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_rerank = enable_rerank

        logger.info(
            f"混合检索器初始化完成: "
            f"top_k={top_k}, rerank_top_n={rerank_top_n}, "
            f"向量权重={vector_weight}, BM25权重={bm25_weight}, "
            f"启用重排={enable_rerank}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievalResult], RetrievalStats]:
        """
        执行混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量（可选，覆盖默认值）
            filter: 元数据过滤条件

        Returns:
            (检索结果列表, 检索统计信息)
        """
        import time
        start_time = time.time()

        top_k = top_k or self.top_k
        stats = RetrievalStats(query=query)

        # 1. 向量检索
        vector_results = self._vector_search(query, top_k, filter)
        stats.vector_count = len(vector_results)
        stats.total_candidates += len(vector_results)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, top_k)
        stats.bm25_count = len(bm25_results)
        stats.total_candidates += len(bm25_results)

        # 3. 合并结果
        merged_results = self._merge_results(vector_results, bm25_results)
        stats.merged_count = len(merged_results)

        # 4. Rerank 精排
        if self.enable_rerank and self.reranker is not None and merged_results:
            final_results = self._rerank(query, merged_results)
            stats.reranked_count = len(final_results)
        else:
            # 不重排，直接取 top_k
            final_results = merged_results[:top_k]
            for i, result in enumerate(final_results, start=1):
                result.rank = i

        stats.final_count = len(final_results)
        stats.latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"混合检索完成: 查询='{query[:30]}...', "
            f"候选={stats.total_candidates}, "
            f"合并={stats.merged_count}, "
            f"最终={stats.final_count}, "
            f"耗时={stats.latency_ms:.1f}ms"
        )

        return final_results, stats

    def _vector_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 返回数量
            filter: 过滤条件

        Returns:
            检索结果列表
        """
        if self.vector_store is None:
            return []

        try:
            # 带分数的相似度搜索
            results_with_score = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter,
            )

            retrieval_results = []
            for rank, (doc, score) in enumerate(results_with_score, start=1):
                # ChromaDB 的分数是距离，需要转换为相似度（越小越相似 -> 越大越相似）
                # 余弦距离范围 [0, 2]，转换为相似度 [0, 1]
                similarity = 1.0 - score / 2.0 if score <= 2 else 0.0

                retrieval_results.append(RetrievalResult(
                    document=doc,
                    score=similarity,
                    rank=rank,
                    source="vector",
                    vector_score=similarity,
                ))

            return retrieval_results

        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if self.bm25_retriever is None:
            return []

        try:
            bm25_results = self.bm25_retriever.search(query, top_k=top_k)

            if not bm25_results:
                return []

            # 归一化 BM25 分数到 [0, 1]
            max_score = max(r.score for r in bm25_results) if bm25_results else 1
            if max_score == 0:
                max_score = 1

            retrieval_results = []
            for result in bm25_results:
                normalized_score = result.score / max_score

                retrieval_results.append(RetrievalResult(
                    document=result.document,
                    score=normalized_score,
                    rank=result.rank,
                    source="bm25",
                    bm25_score=normalized_score,
                ))

            return retrieval_results

        except Exception as e:
            logger.error(f"BM25 检索失败: {str(e)}")
            return []

    def _merge_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        合并向量检索和 BM25 检索结果

        合并策略：
            - 去重：相同文档合并
            - 加权融合：向量分数 * 向量权重 + BM25分数 * BM25权重
            - 排序：按融合分数降序排列

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果

        Returns:
            合并后的结果列表
        """
        # 使用文档内容作为去重键
        merged = {}

        # 添加向量检索结果
        for result in vector_results:
            key = result.document.page_content
            if key not in merged:
                merged[key] = RetrievalResult(
                    document=result.document,
                    score=0.0,
                    rank=0,
                    source="hybrid",
                    vector_score=result.vector_score,
                    bm25_score=None,
                )
            else:
                merged[key].vector_score = result.vector_score

        # 添加 BM25 检索结果
        for result in bm25_results:
            key = result.document.page_content
            if key not in merged:
                merged[key] = RetrievalResult(
                    document=result.document,
                    score=0.0,
                    rank=0,
                    source="hybrid",
                    vector_score=None,
                    bm25_score=result.bm25_score,
                )
            else:
                merged[key].bm25_score = result.bm25_score

        # 计算融合分数
        for key, result in merged.items():
            vector_score = result.vector_score or 0.0
            bm25_score = result.bm25_score or 0.0

            # 加权融合
            result.score = (
                vector_score * self.vector_weight +
                bm25_score * self.bm25_weight
            )

            # 如果只有一种检索结果，适当降低权重（避免单一来源分数过高）
            if result.vector_score is None:
                result.score = bm25_score * self.bm25_weight
            elif result.bm25_score is None:
                result.score = vector_score * self.vector_weight

        # 排序
        sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)

        # 更新排名
        for rank, result in enumerate(sorted_results, start=1):
            result.rank = rank

        return sorted_results

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Rerank 精排

        Args:
            query: 查询文本
            results: 待重排的结果

        Returns:
            重排后的结果列表
        """
        if not results or self.reranker is None:
            return results

        try:
            # 提取文档
            documents = [r.document for r in results]

            # 重排
            rerank_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=self.rerank_top_n,
            )

            # 构建最终结果
            final_results = []
            for rerank_result in rerank_results:
                # 找到原始结果以保留其他信息
                original = None
                for r in results:
                    if r.document.page_content == rerank_result.document.page_content:
                        original = r
                        break

                if original:
                    original.rerank_score = rerank_result.score
                    original.score = rerank_result.score  # 使用重排分数作为最终分数
                    original.rank = rerank_result.new_rank
                    original.source = "hybrid_reranked"
                    final_results.append(original)
                else:
                    final_results.append(RetrievalResult(
                        document=rerank_result.document,
                        score=rerank_result.score,
                        rank=rerank_result.new_rank,
                        source="hybrid_reranked",
                        rerank_score=rerank_result.score,
                    ))

            return final_results

        except Exception as e:
            logger.error(f"重排失败: {str(e)}，返回原始结果")
            return results[:self.rerank_top_n]

    def update_bm25_index(self, documents: List[Document]) -> None:
        """
        更新 BM25 索引

        Args:
            documents: 新增的文档列表
        """
        if self.bm25_retriever is not None:
            self.bm25_retriever.add_documents(documents)
            logger.info(f"BM25 索引已更新，新增 {len(documents)} 个文档")

    def rebuild_bm25_index(self, documents: List[Document]) -> None:
        """
        重建 BM25 索引

        Args:
            documents: 所有文档列表
        """
        if self.bm25_retriever is not None:
            self.bm25_retriever.clear()
            self.bm25_retriever.add_documents(documents)
            logger.info(f"BM25 索引已重建，共 {len(documents)} 个文档")
