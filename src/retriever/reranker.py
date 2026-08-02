"""
Rerank 重排模块 - 检索结果精排优化
Rerank Module - Retrieval Result Refinement

功能说明：
    对初步检索结果进行重新排序，使用更精细的模型计算查询与文档的相关性，
    显著提升检索准确率。支持本地 BGE-Reranker 和 Cohere Rerank API 两种方案。

技术选型：
    - BGE-Reranker：北京智源研究院开源，中文效果优秀，本地部署
    - Cohere Rerank：商业 API，效果极佳，适合生产环境
    - 交叉编码器：直接计算 query-document 对的相似度

优化策略：
    1. 精排优化：从 top_k 中选出 top_n，提升准确率
    2. 分数归一化：统一不同检索方式的分数范围
    3. 批量处理：批量计算相似度，提升效率
    4. 降级策略：重排失败时回退到原始排序
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from loguru import logger
from langchain_core.documents import Document


@dataclass
class RerankResult:
    """重排结果"""
    document: Document
    score: float
    original_rank: int
    new_rank: int


class Reranker:
    """
    重排器基类
    """

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """
        重排文档

        Args:
            query: 查询文本
            documents: 待重排的文档列表
            top_n: 返回前 N 个结果

        Returns:
            重排后的结果列表
        """
        raise NotImplementedError


class BGEReranker(Reranker):
    """
    BGE Reranker 本地重排器

    使用 BAAI/bge-reranker 系列模型进行本地重排，
    适合数据不出域的场景。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
        use_fp16: bool = False,
    ):
        """
        初始化 BGE 重排器

        Args:
            model_name: 模型名称
            device: 运行设备（cpu/cuda）
            use_fp16: 是否使用半精度
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None

        self._init_model()
        logger.info(f"BGE Reranker 初始化完成: 模型={model_name}, 设备={device}")

    def _init_model(self) -> None:
        """初始化模型"""
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
            )
            logger.info("BGE Reranker 模型加载成功")
        except ImportError:
            logger.warning("FlagEmbedding 未安装，BGE Reranker 不可用")
            self._model = None
        except Exception as e:
            logger.warning(f"BGE Reranker 加载失败: {str(e)}")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """
        使用 BGE 模型重排文档

        Args:
            query: 查询文本
            documents: 待重排的文档列表
            top_n: 返回前 N 个结果

        Returns:
            重排后的结果列表
        """
        if not documents:
            return []

        if self._model is None:
            logger.warning("BGE Reranker 未初始化，返回原始顺序")
            return self._fallback_rerank(documents, top_n)

        try:
            # 构建 query-document 对
            pairs = [[query, doc.page_content] for doc in documents]

            # 计算相似度分数
            scores = self._model.compute_score(pairs, normalize=True)

            # 确保 scores 是列表
            if not isinstance(scores, list):
                scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

            # 排序
            scored_docs = list(zip(documents, scores, range(1, len(documents) + 1)))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 生成结果
            results = []
            for new_rank, (doc, score, orig_rank) in enumerate(scored_docs[:top_n], start=1):
                results.append(RerankResult(
                    document=doc,
                    score=float(score),
                    original_rank=orig_rank,
                    new_rank=new_rank,
                ))

            logger.debug(f"BGE 重排完成: {len(documents)} -> {len(results)}")
            return results

        except Exception as e:
            logger.error(f"BGE 重排失败: {str(e)}，回退到原始顺序")
            return self._fallback_rerank(documents, top_n)

    def _fallback_rerank(
        self,
        documents: List[Document],
        top_n: int,
    ) -> List[RerankResult]:
        """
        降级重排（保持原始顺序）

        Args:
            documents: 文档列表
            top_n: 返回数量

        Returns:
            重排结果列表
        """
        results = []
        for rank, doc in enumerate(documents[:top_n], start=1):
            results.append(RerankResult(
                document=doc,
                score=1.0 - rank * 0.01,  # 伪分数
                original_rank=rank,
                new_rank=rank,
            ))
        return results


class CohereReranker(Reranker):
    """
    Cohere Rerank API 重排器

    使用 Cohere 官方 Rerank API 进行重排，效果极佳，
    适合生产环境使用。
    """

    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0"):
        """
        初始化 Cohere 重排器

        Args:
            api_key: Cohere API 密钥
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model
        self._client = None

        if api_key:
            self._init_client()
        else:
            logger.warning("未提供 Cohere API 密钥，Cohere Reranker 不可用")

        logger.info(f"Cohere Reranker 初始化完成: 模型={model}")

    def _init_client(self) -> None:
        """初始化 Cohere 客户端"""
        try:
            import cohere
            self._client = cohere.ClientV2(self.api_key)
            logger.info("Cohere 客户端初始化成功")
        except ImportError:
            logger.warning("cohere 库未安装")
            self._client = None
        except Exception as e:
            logger.warning(f"Cohere 客户端初始化失败: {str(e)}")
            self._client = None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """
        使用 Cohere API 重排文档

        Args:
            query: 查询文本
            documents: 待重排的文档列表
            top_n: 返回前 N 个结果

        Returns:
            重排后的结果列表
        """
        if not documents:
            return []

        if self._client is None:
            logger.warning("Cohere 客户端未初始化，返回原始顺序")
            return self._fallback_rerank(documents, top_n)

        try:
            # 提取文档文本
            docs_text = [doc.page_content for doc in documents]

            # 调用 Cohere Rerank API
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=docs_text,
                top_n=top_n,
            )

            # 解析结果
            results = []
            for new_rank, result in enumerate(response.results, start=1):
                orig_idx = result.index
                doc = documents[orig_idx]
                results.append(RerankResult(
                    document=doc,
                    score=result.relevance_score,
                    original_rank=orig_idx + 1,
                    new_rank=new_rank,
                ))

            logger.debug(f"Cohere 重排完成: {len(documents)} -> {len(results)}")
            return results

        except Exception as e:
            logger.error(f"Cohere 重排失败: {str(e)}，回退到原始顺序")
            return self._fallback_rerank(documents, top_n)

    def _fallback_rerank(
        self,
        documents: List[Document],
        top_n: int,
    ) -> List[RerankResult]:
        """降级重排"""
        results = []
        for rank, doc in enumerate(documents[:top_n], start=1):
            results.append(RerankResult(
                document=doc,
                score=1.0 - rank * 0.01,
                original_rank=rank,
                new_rank=rank,
            ))
        return results


def create_reranker(
    provider: str = "bge",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Reranker:
    """
    工厂函数：创建重排器

    Args:
        provider: 重排器类型，可选 bge/cohere
        model_name: 模型名称
        api_key: API 密钥（cohere 需要）

    Returns:
        Reranker 实例
    """
    if provider == "bge":
        return BGEReranker(model_name=model_name or "BAAI/bge-reranker-large")
    elif provider == "cohere":
        return CohereReranker(api_key=api_key or "", model=model_name or "rerank-multilingual-v3.0")
    else:
        raise ValueError(f"不支持的重排器类型: {provider}")
