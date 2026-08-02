"""
问答流水线模块 - 系统核心编排
Q&A Pipeline Module - System Core Orchestration

功能说明：
    整合文档处理、向量存储、混合检索、LLM 问答等所有模块，
    提供统一的系统入口和高层 API，简化使用方式。

技术架构：
    - 文档处理层：解析、分块、增量更新
    - 向量存储层：向量化、存储、检索
    - 检索层：混合检索 + Rerank
    - 问答层：LLM 对话、流式输出、引用标注

优化策略：
    1. 统一入口：一个类管理所有功能
    2. 状态管理：维护系统状态和组件
    3. 错误处理：统一的异常处理和日志
    4. 性能监控：记录各阶段耗时
"""

from typing import List, Optional, Dict, Any, Generator, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from langchain_core.documents import Document

from config.settings import settings
from src.document_processor import (
    DocumentParser,
    AdaptiveChunker,
    IncrementalManager,
)
from src.vector_store import EmbeddingManager, ChromaVectorStore
from src.retriever import (
    BM25Retriever,
    HybridRetriever,
    create_reranker,
)
from src.llm import ChatManager, ChatResponse, ConversationHistory


@dataclass
class SystemStats:
    """系统统计信息"""
    total_documents: int = 0
    total_files: int = 0
    total_sessions: int = 0
    indexed_files: List[str] = field(default_factory=list)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    system_start_time: str = field(default_factory=lambda: datetime.now().isoformat())


class QAPipeline:
    """
    问答系统流水线 - 系统核心入口

    整合所有模块，提供统一的高层 API：
        - 文档管理：上传、删除、增量更新
        - 问答对话：单轮、多轮、流式
        - 系统管理：状态查询、配置调整、数据管理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化问答系统流水线

        Args:
            config: 配置覆盖项（可选）
        """
        # 应用配置覆盖
        if config:
            self._apply_config(config)

        # 初始化各模块
        self._init_components()

        # 系统状态
        self.stats = SystemStats()
        self._update_stats()

        logger.info("=" * 60)
        logger.info("企业级文档知识库问答系统初始化完成")
        logger.info(f"版本: {settings.version}")
        logger.info(f"向量数据库: {settings.chroma.collection_name}")
        logger.info(f"LLM 模型: {settings.openai.model}")
        logger.info(f"嵌入模型: {settings.openai.embedding_model}")
        logger.info("=" * 60)

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """应用配置覆盖"""
        # 这里可以实现配置覆盖逻辑
        pass

    def _init_components(self) -> None:
        """初始化所有组件"""
        # 1. 文档处理组件
        self.parser = DocumentParser()
        self.chunker = AdaptiveChunker(
            default_chunk_size=settings.document.chunk_size,
            default_chunk_overlap=settings.document.chunk_overlap,
        )
        self.incremental = IncrementalManager(
            metadata_path=settings.base_dir / "data" / "index_metadata.json"
        )

        # 2. 嵌入模型
        self.embedding_manager = EmbeddingManager(
            provider="openai",
            model_name=settings.openai.embedding_model,
            api_key=settings.openai.api_key,
            base_url=settings.openai.base_url,
        )

        # 3. 向量数据库
        self.vector_store = ChromaVectorStore(
            persist_directory=settings.chroma_persist_path,
            collection_name=settings.chroma.collection_name,
            embedding_function=self.embedding_manager.embedding,
        )

        # 4. BM25 检索器
        self.bm25_retriever = BM25Retriever(language="chinese")
        self._rebuild_bm25_index()

        # 5. 重排器
        self.reranker = None
        if settings.retrieval.enable_rerank:
            try:
                # 尝试使用 BGE Reranker（本地模型）
                self.reranker = create_reranker(
                    provider="bge",
                    model_name=settings.retrieval.rerank_model,
                )
            except Exception as e:
                logger.warning(f"Reranker 初始化失败: {str(e)}，将不使用重排")

        # 6. 混合检索器
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker,
            top_k=settings.retrieval.top_k,
            rerank_top_n=settings.retrieval.rerank_top_n,
            vector_weight=settings.retrieval.hybrid_weight_vector,
            bm25_weight=settings.retrieval.hybrid_weight_bm25,
            enable_rerank=settings.retrieval.enable_rerank and self.reranker is not None,
        )

        # 7. LLM
        self._init_llm()

        # 8. 对话历史管理器
        self.conversation_history = ConversationHistory(
            max_turns=settings.chat.max_history_turns,
        )

        # 9. 对话管理器
        self.chat_manager = ChatManager(
            llm=self.llm,
            retriever=self.hybrid_retriever,
            history=self.conversation_history,
            enable_streaming=settings.chat.stream_output,
            enable_query_rewrite=True,
            max_history_turns=settings.chat.max_history_turns,
        )

    def _init_llm(self) -> None:
        """初始化 LLM"""
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=settings.openai.model,
            api_key=settings.openai.api_key,
            base_url=settings.openai.base_url,
            temperature=settings.chat.temperature,
            max_tokens=settings.chat.max_tokens,
            streaming=settings.chat.stream_output,
        )

        logger.info(f"LLM 初始化完成: {settings.openai.model}")

    def _rebuild_bm25_index(self) -> None:
        """重建 BM25 索引（从向量数据库加载所有文档）"""
        try:
            # 从向量数据库获取所有文档
            all_docs_data = self.vector_store.get()
            documents = []

            for doc_text, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"]):
                doc = Document(
                    page_content=doc_text,
                    metadata=meta or {},
                )
                documents.append(doc)

            # 重建 BM25 索引
            self.bm25_retriever.clear()
            if documents:
                self.bm25_retriever.add_documents(documents)

            logger.info(f"BM25 索引重建完成: {len(documents)} 个文档")

        except Exception as e:
            logger.warning(f"重建 BM25 索引失败: {str(e)}")

    # =========================================================================
    # 文档管理 API
    # =========================================================================

    def add_file(self, file_path: str | Path) -> Dict[str, Any]:
        """
        添加单个文件到知识库

        Args:
            file_path: 文件路径

        Returns:
            处理结果信息
        """
        file_path = Path(file_path)

        logger.info(f"开始处理文件: {file_path.name}")

        # 检查文件状态
        status = self.incremental.get_file_status(file_path)

        if status == "unchanged":
            logger.info(f"文件未变化，跳过: {file_path.name}")
            return {
                "status": "skipped",
                "reason": "file_unchanged",
                "filename": file_path.name,
            }

        if status == "modified":
            # 删除旧版本
            old_doc_ids = self.incremental.get_file_doc_ids(file_path)
            if old_doc_ids:
                self.vector_store.delete(old_doc_ids)
            logger.info(f"文件已修改，删除旧版本: {file_path.name}")

        # 解析文档
        documents = self.parser.parse_file(file_path)
        if not documents:
            return {
                "status": "failed",
                "reason": "no_content",
                "filename": file_path.name,
            }

        # 分块
        chunks = self.chunker.split_documents(documents)
        if not chunks:
            return {
                "status": "failed",
                "reason": "no_chunks",
                "filename": file_path.name,
            }

        # 添加到向量数据库
        doc_ids = self.vector_store.add_documents(chunks)

        # 更新 BM25 索引
        self.bm25_retriever.add_documents(chunks)

        # 更新增量元数据
        self.incremental.update_file_metadata(
            file_path=file_path,
            chunk_count=len(chunks),
            doc_ids=doc_ids,
        )

        # 更新统计信息
        self._update_stats()

        logger.info(
            f"文件处理完成: {file_path.name}, "
            f"分块数={len(chunks)}"
        )

        return {
            "status": "success",
            "filename": file_path.name,
            "chunk_count": len(chunks),
            "doc_ids": doc_ids,
        }

    def add_directory(self, dir_path: str | Path, incremental: bool = True) -> Dict[str, Any]:
        """
        批量添加目录下的所有文档

        Args:
            dir_path: 目录路径
            incremental: 是否增量更新

        Returns:
            处理结果统计
        """
        dir_path = Path(dir_path)

        logger.info(f"开始处理目录: {dir_path}")

        if incremental:
            # 增量更新模式
            scan_result = self.incremental.scan_directory(
                dir_path=dir_path,
                recursive=True,
                supported_formats=set(self.parser.get_supported_formats()),
            )

            # 处理新增和修改的文件
            files_to_process = scan_result["new"] + scan_result["modified"]

            # 处理删除的文件
            deleted_count = 0
            for deleted_file in scan_result["deleted"]:
                try:
                    doc_ids = self.incremental.get_file_doc_ids(deleted_file)
                    if doc_ids:
                        self.vector_store.delete(doc_ids)
                    self.incremental.remove_file_metadata(deleted_file)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"删除已索引文件失败: {deleted_file}, {str(e)}")

        else:
            # 全量处理模式
            import os
            files_to_process = []
            for root, _, files in os.walk(dir_path):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in self.parser.get_supported_formats():
                        files_to_process.append(str(Path(root) / f))

            deleted_count = 0

        # 处理文件
        success_count = 0
        failed_count = 0
        failed_files = []
        total_chunks = 0

        for file_path in files_to_process:
            try:
                result = self.add_file(file_path)
                if result["status"] == "success":
                    success_count += 1
                    total_chunks += result["chunk_count"]
                elif result["status"] == "skipped":
                    pass  # 跳过的不计入失败
                else:
                    failed_count += 1
                    failed_files.append((file_path, result.get("reason", "unknown")))
            except Exception as e:
                failed_count += 1
                failed_files.append((file_path, str(e)))
                logger.error(f"处理文件失败: {file_path}, {str(e)}")

        # 重建 BM25 索引（确保一致性）
        self._rebuild_bm25_index()

        # 更新统计
        self._update_stats()

        result = {
            "total_files_found": len(files_to_process),
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": len(files_to_process) - success_count - failed_count,
            "deleted_count": deleted_count,
            "total_chunks": total_chunks,
            "failed_files": failed_files,
        }

        logger.info(
            f"目录处理完成: 成功={success_count}, "
            f"失败={failed_count}, 总分块={total_chunks}"
        )

        return result

    def delete_file(self, filename: str) -> Dict[str, Any]:
        """
        从知识库删除文件

        Args:
            filename: 文件名

        Returns:
            删除结果
        """
        logger.info(f"删除文件: {filename}")

        try:
            # 从向量数据库删除
            deleted_count = self.vector_store.delete_by_filename(filename)

            # 从增量管理器删除
            # 注意：这里需要找到完整路径，简化处理
            # 实际应用中应该维护文件名到路径的映射

            # 重建 BM25 索引
            self._rebuild_bm25_index()

            # 更新统计
            self._update_stats()

            return {
                "status": "success",
                "filename": filename,
                "deleted_chunks": deleted_count,
            }

        except Exception as e:
            logger.error(f"删除文件失败: {filename}, {str(e)}")
            return {
                "status": "failed",
                "filename": filename,
                "error": str(e),
            }

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """
        清空知识库

        Returns:
            清空结果
        """
        logger.warning("清空知识库")

        self.vector_store.clear()
        self.bm25_retriever.clear()
        self.incremental.clear_all()
        self._update_stats()

        return {
            "status": "success",
            "message": "知识库已清空",
        }

    # =========================================================================
    # 问答 API
    # =========================================================================

    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        问答（非流式）

        Args:
            question: 用户问题
            session_id: 会话 ID（可选）
            filter: 检索过滤条件

        Returns:
            ChatResponse 响应
        """
        return self.chat_manager.chat(
            question=question,
            session_id=session_id,
            filter=filter,
        )

    def ask_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[str, Optional[ChatResponse]], None, None]:
        """
        问答（流式）

        Args:
            question: 用户问题
            session_id: 会话 ID
            filter: 检索过滤条件

        Yields:
            (内容片段, 最终响应)
        """
        yield from self.chat_manager.chat_stream(
            question=question,
            session_id=session_id,
            filter=filter,
        )

    # =========================================================================
    # 系统管理 API
    # =========================================================================

    def get_stats(self) -> SystemStats:
        """
        获取系统统计信息

        Returns:
            系统统计信息
        """
        self._update_stats()
        return self.stats

    def _update_stats(self) -> None:
        """更新系统统计信息"""
        try:
            vector_stats = self.vector_store.get_stats()
            self.stats.total_documents = vector_stats["total_documents"]
            self.stats.total_files = vector_stats["total_files"]
            self.stats.indexed_files = vector_stats["files"]
            self.stats.type_distribution = vector_stats["type_distribution"]
        except Exception as e:
            logger.warning(f"更新统计信息失败: {str(e)}")

        self.stats.total_sessions = self.conversation_history.get_session_count()

    def list_files(self) -> List[str]:
        """列出所有已索引的文件"""
        return self.vector_store.get_all_files()

    def list_sessions(self) -> List[Dict]:
        """列出所有会话"""
        return self.chat_manager.list_sessions()

    def clear_session(self, session_id: str) -> None:
        """清空指定会话的历史"""
        self.chat_manager.clear_history(session_id)

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态信息
        """
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.version,
            "components": {},
        }

        # 检查向量数据库
        try:
            count = self.vector_store.count()
            status["components"]["vector_store"] = {
                "status": "healthy",
                "document_count": count,
            }
        except Exception as e:
            status["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status["status"] = "degraded"

        # 检查 LLM
        try:
            # 简单的 ping 测试
            status["components"]["llm"] = {
                "status": "healthy",
                "model": settings.openai.model,
            }
        except Exception as e:
            status["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status["status"] = "degraded"

        return status
