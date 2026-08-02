"""
智能分块模块 - 多种分块策略支持
Smart Chunker Module - Multiple Chunking Strategies

功能说明：
    提供多种文档分块策略，确保分块后的文本片段既保持语义完整性，
    又满足向量检索的粒度要求。支持固定大小分块、语义分块、递归分块等多种策略。

技术选型：
    - LangChain TextSplitter 系列：成熟的分块工具
    - tiktoken：精确的 token 计数，适配 OpenAI 模型
    - 自定义语义分块：基于段落、标题等结构信息

优化策略：
    1. 递归字符分块：优先在段落、句子、单词边界分割
    2. 重叠分块：相邻块之间保留重叠，避免上下文断裂
    3. 元数据继承：分块后保留原始文档的所有元数据
    4. 块大小自适应：根据文档类型调整分块参数
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    SpacyTextSplitter,
)
from loguru import logger


class DocumentChunker:
    """
    智能文档分块器

    支持的分块策略：
        - recursive: 递归字符分块（推荐，默认）
        - token: 基于 token 数的分块
        - markdown: Markdown 标题分块
        - sentence: 基于句子的分块（需要 spaCy）
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "recursive",
        separators: Optional[List[str]] = None,
    ):
        """
        初始化文档分块器

        Args:
            chunk_size: 分块大小（字符数或 token 数，取决于策略）
            chunk_overlap: 分块重叠大小
            strategy: 分块策略，可选 recursive/token/markdown/sentence
            separators: 自定义分隔符列表（仅 recursive 策略有效）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        # 默认分隔符（按优先级排序）
        if separators is None:
            self.separators = [
                "\n\n",   # 段落
                "\n",     # 行
                "。",     # 中文句号
                "！",     # 中文感叹号
                "？",     # 中文问号
                ". ",     # 英文句号
                "! ",     # 英文感叹号
                "? ",     # 英文问号
                "；",     # 中文分号
                "; ",     # 英文分号
                "，",     # 中文逗号
                ", ",     # 英文逗号
                " ",      # 空格
                "",       # 字符
            ]
        else:
            self.separators = separators

        self._splitter = self._init_splitter()
        logger.info(
            f"文档分块器初始化完成: 策略={strategy}, "
            f"块大小={chunk_size}, 重叠={chunk_overlap}"
        )

    def _init_splitter(self):
        """初始化分块器"""
        if self.strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
            )
        elif self.strategy == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base",  # GPT-4/GPT-3.5 的编码
            )
        elif self.strategy == "markdown":
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False,
            )
        elif self.strategy == "sentence":
            try:
                return SpacyTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    pipeline="zh_core_web_sm",
                )
            except Exception:
                logger.warning("spaCy 中文模型未安装，回退到递归分块策略")
                return RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators,
                )
        else:
            raise ValueError(f"不支持的分块策略: {self.strategy}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        批量分块文档

        Args:
            documents: 待分块的 Document 列表

        Returns:
            分块后的 Document 列表
        """
        if not documents:
            return []

        logger.info(f"开始分块 {len(documents)} 个文档...")

        all_chunks = []
        for doc in documents:
            try:
                chunks = self._split_single_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"文档分块失败: {doc.metadata.get('filename', 'unknown')}, 错误: {str(e)}")
                # 分块失败时保留原始文档
                all_chunks.append(doc)

        logger.info(
            f"分块完成: 原始 {len(documents)} 个文档 -> {len(all_chunks)} 个分块"
        )

        return all_chunks

    def _split_single_document(self, document: Document) -> List[Document]:
        """
        分块单个文档

        Args:
            document: 待分块的 Document

        Returns:
            分块后的 Document 列表
        """
        # Markdown 策略特殊处理
        if self.strategy == "markdown":
            chunks = self._splitter.split_text(document.page_content)
            # 合并元数据
            for chunk in chunks:
                chunk.metadata.update(document.metadata)
            return chunks

        # 其他策略
        chunks = self._splitter.split_documents([document])

        # 为每个分块添加索引信息
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        直接分块纯文本

        Args:
            text: 待分块的文本

        Returns:
            分块后的文本列表
        """
        if not text.strip():
            return []

        if self.strategy == "markdown":
            chunks = self._splitter.split_text(text)
            return [chunk.page_content for chunk in chunks]
        else:
            return self._splitter.split_text(text)


class AdaptiveChunker(DocumentChunker):
    """
    自适应分块器 - 根据文档类型自动选择分块策略

    优化策略：
        - PDF 文档：使用较大的块大小（页面内容较多）
        - Word 文档：使用递归分块（段落结构清晰）
        - Markdown 文档：使用标题分块（结构明确）
        - Excel/CSV：保持原始行粒度（每行信息独立）
        - 纯文本：使用标准分块
    """

    def __init__(
        self,
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 64,
    ):
        """
        初始化自适应分块器

        Args:
            default_chunk_size: 默认分块大小
            default_chunk_overlap: 默认分块重叠大小
        """
        super().__init__(
            chunk_size=default_chunk_size,
            chunk_overlap=default_chunk_overlap,
            strategy="recursive",
        )
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self._chunkers = {}

    def _get_chunker_for_file_type(self, file_type: str) -> DocumentChunker:
        """
        根据文件类型获取对应的分块器

        Args:
            file_type: 文件类型（pdf/docx/markdown 等）

        Returns:
            对应的 DocumentChunker 实例
        """
        if file_type in self._chunkers:
            return self._chunkers[file_type]

        # 根据文件类型配置不同的分块参数
        config_map = {
            "pdf": {
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap,
                "strategy": "recursive",
            },
            "docx": {
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap,
                "strategy": "recursive",
            },
            "pptx": {
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap // 2,
                "strategy": "recursive",
            },
            "markdown": {
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap,
                "strategy": "markdown",
            },
            "xlsx": {
                "chunk_size": self.default_chunk_size * 2,  # Excel 行内容通常较短
                "chunk_overlap": 0,
                "strategy": "recursive",
            },
            "csv": {
                "chunk_size": self.default_chunk_size * 2,
                "chunk_overlap": 0,
                "strategy": "recursive",
            },
            "text": {
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap,
                "strategy": "recursive",
            },
            "html": {
                "chunk_size": self.default_chunk_size,
                "chunk_overlap": self.default_chunk_overlap,
                "strategy": "recursive",
            },
        }

        config = config_map.get(file_type, config_map["text"])
        chunker = DocumentChunker(**config)
        self._chunkers[file_type] = chunker

        return chunker

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        自适应分块 - 按文件类型分组后分别分块

        Args:
            documents: 待分块的 Document 列表

        Returns:
            分块后的 Document 列表
        """
        if not documents:
            return []

        # 按文件类型分组
        type_groups = {}
        for doc in documents:
            file_type = doc.metadata.get("file_type", "text")
            if file_type not in type_groups:
                type_groups[file_type] = []
            type_groups[file_type].append(doc)

        logger.info(f"自适应分块: 发现 {len(type_groups)} 种文件类型")

        all_chunks = []
        for file_type, docs in type_groups.items():
            logger.info(f"处理 {file_type} 类型文档: {len(docs)} 个")
            chunker = self._get_chunker_for_file_type(file_type)
            chunks = chunker.split_documents(docs)
            all_chunks.extend(chunks)

        logger.info(f"自适应分块完成: 共 {len(all_chunks)} 个分块")
        return all_chunks
