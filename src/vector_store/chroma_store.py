"""
ChromaDB 向量数据库模块 - 向量存储与检索
ChromaDB Vector Store Module - Vector Storage and Retrieval

功能说明：
    基于 ChromaDB 的向量数据库实现，提供文档的向量化存储、相似度检索、
    条件过滤等功能。支持持久化存储，确保数据不丢失。

技术选型：
    - ChromaDB：轻量级、本地部署、零配置，适合企业级知识库
    - 持久化存储：数据保存到磁盘，重启不丢失
    - 元数据过滤：支持按文件、类型等条件过滤检索结果

优化策略：
    1. 批量插入：批量添加文档，提升效率
    2. 元数据索引：支持按文件、类型等多维度过滤
    3. 增量更新：支持单文档/批量文档的增删改
    4. 相似度搜索：支持余弦相似度、欧氏距离等多种度量
"""

import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger


class ChromaVectorStore:
    """
    ChromaDB 向量数据库封装

    功能：
        - 文档向量化存储
        - 相似度检索
        - 元数据过滤检索
        - 文档增删改查
        - 集合管理
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "enterprise_docs",
        embedding_function=None,
    ):
        """
        初始化 ChromaDB 向量存储

        Args:
            persist_directory: 持久化存储目录
            collection_name: 集合名称
            embedding_function: 嵌入函数
        """
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self._embedding_function = embedding_function
        self._client = None
        self._collection = None
        self._langchain_vectorstore = None

        self._init_chroma()
        logger.info(
            f"ChromaDB 初始化完成: 集合={collection_name}, "
            f"存储目录={persist_directory}"
        )

    def _init_chroma(self) -> None:
        """初始化 ChromaDB 客户端和集合"""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from langchain_community.vectorstores import Chroma

        # 确保目录存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # 创建 ChromaDB 客户端
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )

        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

        # 创建 LangChain Chroma 实例（用于兼容 LangChain 接口）
        if self._embedding_function is not None:
            self._langchain_vectorstore = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self._embedding_function,
            )

    @property
    def langchain_store(self):
        """获取 LangChain 兼容的向量存储实例"""
        if self._langchain_vectorstore is None:
            raise ValueError("未设置嵌入函数，无法创建 LangChain 实例")
        return self._langchain_vectorstore

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加文档到向量数据库

        Args:
            documents: Document 列表
            ids: 自定义 ID 列表（可选）

        Returns:
            添加的文档 ID 列表
        """
        if not documents:
            return []

        # 生成 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # 提取文本和元数据
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            # 使用 LangChain 接口添加
            if self._langchain_vectorstore is not None:
                self._langchain_vectorstore.add_documents(
                    documents=documents,
                    ids=ids,
                )
            else:
                # 直接使用 ChromaDB 客户端（需要外部提供嵌入向量）
                self._collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                )

            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            return ids

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        直接添加文本到向量数据库

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）
            ids: 自定义 ID 列表（可选）

        Returns:
            添加的文档 ID 列表
        """
        if not texts:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        try:
            if self._langchain_vectorstore is not None:
                self._langchain_vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
            else:
                self._collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas or [{} for _ in texts],
                )

            logger.info(f"成功添加 {len(texts)} 个文本到向量数据库")
            return ids

        except Exception as e:
            logger.error(f"添加文本失败: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            最相似的 Document 列表
        """
        try:
            if self._langchain_vectorstore is not None:
                results = self._langchain_vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                )
                return results
            else:
                # 直接使用 ChromaDB（需要外部提供查询向量）
                raise ValueError("未设置嵌入函数，无法执行相似度搜索")

        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        带分数的相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            (Document, 分数) 元组列表
        """
        try:
            if self._langchain_vectorstore is not None:
                results = self._langchain_vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                )
                return results
            else:
                raise ValueError("未设置嵌入函数，无法执行相似度搜索")

        except Exception as e:
            logger.error(f"带分数相似度搜索失败: {str(e)}")
            raise

    def delete(self, ids: List[str]) -> None:
        """
        按 ID 删除文档

        Args:
            ids: 要删除的文档 ID 列表
        """
        if not ids:
            return

        try:
            self._collection.delete(ids=ids)
            logger.info(f"成功删除 {len(ids)} 个文档")
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            raise

    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """
        按过滤条件删除文档

        Args:
            filter: 元数据过滤条件

        Returns:
            删除的文档数量
        """
        try:
            # 先查询符合条件的文档 ID
            results = self._collection.get(where=filter, include=[])
            ids = results["ids"]

            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"按条件删除 {len(ids)} 个文档")

            return len(ids)

        except Exception as e:
            logger.error(f"按条件删除文档失败: {str(e)}")
            raise

    def delete_by_filename(self, filename: str) -> int:
        """
        按文件名删除所有相关文档

        Args:
            filename: 文件名

        Returns:
            删除的文档数量
        """
        return self.delete_by_filter({"filename": filename})

    def get(self, ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取文档信息

        Args:
            ids: 文档 ID 列表（可选，不传则返回所有）

        Returns:
            包含 ids、documents、metadatas 的字典
        """
        try:
            results = self._collection.get(
                ids=ids,
                include=["documents", "metadatas"],
            )
            return results
        except Exception as e:
            logger.error(f"获取文档失败: {str(e)}")
            raise

    def count(self) -> int:
        """
        获取文档总数

        Returns:
            文档数量
        """
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"获取文档数量失败: {str(e)}")
            raise

    def get_all_files(self) -> List[str]:
        """
        获取所有已索引的文件名列表

        Returns:
            文件名列表
        """
        try:
            # 获取所有元数据
            results = self._collection.get(include=["metadatas"])
            filenames = set()

            for meta in results["metadatas"]:
                if meta and "filename" in meta:
                    filenames.add(meta["filename"])

            return sorted(list(filenames))

        except Exception as e:
            logger.error(f"获取文件列表失败: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息

        Returns:
            统计信息字典
        """
        try:
            total_docs = self.count()
            files = self.get_all_files()

            # 按文件类型统计
            type_counts = {}
            results = self._collection.get(include=["metadatas"])
            for meta in results["metadatas"]:
                if meta and "file_type" in meta:
                    file_type = meta["file_type"]
                    type_counts[file_type] = type_counts.get(file_type, 0) + 1

            return {
                "total_documents": total_docs,
                "total_files": len(files),
                "files": files,
                "type_distribution": type_counts,
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            raise

    def clear(self) -> None:
        """清空集合中的所有数据"""
        try:
            # 删除并重新创建集合
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # 重新创建 LangChain 实例
            if self._embedding_function is not None:
                from langchain_community.vectorstores import Chroma
                self._langchain_vectorstore = Chroma(
                    client=self._client,
                    collection_name=self.collection_name,
                    embedding_function=self._embedding_function,
                )

            logger.info("向量数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
            raise

    def persist(self) -> None:
        """手动持久化（ChromaDB PersistentClient 自动持久化，此方法保留兼容性）"""
        logger.info("ChromaDB 使用持久化客户端，数据自动保存")
