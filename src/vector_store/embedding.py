"""
嵌入模型封装模块 - 统一的向量化接口
Embedding Module - Unified Vectorization Interface

功能说明：
    封装多种嵌入模型，提供统一的向量化接口。支持 OpenAI API 嵌入和本地 Sentence-Transformers 模型，
    可根据配置自动选择合适的嵌入方案。

技术选型：
    - OpenAI Embeddings: 高质量、低延迟，适合生产环境
    - Sentence-Transformers: 本地部署，数据不出域，适合隐私敏感场景
    - BGE 系列：中文场景优化，效果优秀

优化策略：
    1. 批量嵌入：批量处理文本，提升效率
    2. 缓存机制：相同文本直接返回缓存结果
    3. 重试机制：API 调用失败时自动重试
    4. 统一接口：不同模型使用相同的调用方式
"""

from typing import List, Optional
from loguru import logger
from langchain_core.embeddings import Embeddings


class EmbeddingManager:
    """
    嵌入模型管理器

    支持的嵌入模型：
        - openai: OpenAI API 嵌入（默认）
        - local: 本地 Sentence-Transformers 模型
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 32,
        use_cache: bool = True,
    ):
        """
        初始化嵌入管理器

        Args:
            provider: 嵌入模型提供商，可选 openai/local
            model_name: 模型名称
            api_key: API 密钥（openai 提供商需要）
            base_url: API 基础 URL
            batch_size: 批量处理大小
            use_cache: 是否启用缓存
        """
        self.provider = provider
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_cache = use_cache
        self._cache = {}
        self._embedding = None

        if provider == "openai":
            self._init_openai_embeddings(api_key, base_url, model_name)
        elif provider == "local":
            self._init_local_embeddings(model_name)
        else:
            raise ValueError(f"不支持的嵌入提供商: {provider}")

        logger.info(
            f"嵌入模型初始化完成: 提供商={provider}, "
            f"模型={self.model_name}, "
            f"批量大小={batch_size}"
        )

    def _init_openai_embeddings(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model_name: Optional[str],
    ) -> None:
        """初始化 OpenAI 嵌入模型"""
        from langchain_openai import OpenAIEmbeddings

        self.model_name = model_name or "text-embedding-3-small"

        kwargs = {
            "model": self.model_name,
        }

        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._embedding = OpenAIEmbeddings(**kwargs)

    def _init_local_embeddings(self, model_name: Optional[str]) -> None:
        """初始化本地嵌入模型"""
        from langchain_community.embeddings import HuggingFaceEmbeddings

        self.model_name = model_name or "BAAI/bge-small-zh-v1.5"

        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        self._embedding = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    @property
    def embedding(self) -> Embeddings:
        """获取嵌入模型实例"""
        return self._embedding

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        # 检查缓存
        if self.use_cache and text in self._cache:
            return self._cache[text]

        try:
            vector = self._embedding.embed_query(text)

            # 写入缓存
            if self.use_cache:
                self._cache[text] = vector

            return vector
        except Exception as e:
            logger.error(f"嵌入查询文本失败: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []

        # 检查缓存，分离已缓存和未缓存的文本
        cached_results = {}
        uncached_texts = []

        if self.use_cache:
            for text in texts:
                if text in self._cache:
                    cached_results[text] = self._cache[text]
                else:
                    uncached_texts.append(text)
        else:
            uncached_texts = texts

        # 处理未缓存的文本
        if uncached_texts:
            try:
                # 分批处理
                all_vectors = []
                for i in range(0, len(uncached_texts), self.batch_size):
                    batch = uncached_texts[i:i + self.batch_size]
                    vectors = self._embedding.embed_documents(batch)
                    all_vectors.extend(vectors)

                # 写入缓存
                if self.use_cache:
                    for text, vector in zip(uncached_texts, all_vectors):
                        self._cache[text] = vector

                cached_vectors = all_vectors
            except Exception as e:
                logger.error(f"批量嵌入文档失败: {str(e)}")
                raise
        else:
            cached_vectors = []

        # 按原始顺序组装结果
        result = []
        uncached_idx = 0
        for text in texts:
            if text in cached_results:
                result.append(cached_results[text])
            else:
                result.append(cached_vectors[uncached_idx])
                uncached_idx += 1

        return result

    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度

        Returns:
            向量维度
        """
        # 用一个简单的文本测试维度
        test_vector = self.embed_query("test")
        return len(test_vector)

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("嵌入缓存已清空")

    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
