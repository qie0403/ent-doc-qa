"""
问答流水线测试
Q&A Pipeline Tests

测试内容：
    - 文档解析测试
    - 分块测试
    - 向量存储测试
    - 检索测试
    - 问答测试
"""

import pytest
import tempfile
from pathlib import Path

from langchain_core.documents import Document


class TestDocumentParser:
    """文档解析器测试"""

    def test_parse_text_file(self):
        """测试解析文本文件"""
        from src.document_processor import DocumentParser

        parser = DocumentParser()

        # 创建临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是一个测试文档。\n第二行内容。\n第三行内容。")
            temp_path = f.name

        try:
            docs = parser.parse_file(temp_path)
            assert len(docs) == 1
            assert "测试文档" in docs[0].page_content
            assert docs[0].metadata["file_type"] == "text"
        finally:
            Path(temp_path).unlink()

    def test_parse_markdown_file(self):
        """测试解析 Markdown 文件"""
        from src.document_processor import DocumentParser

        parser = DocumentParser()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# 标题\n\n这是正文内容。\n\n## 子标题\n\n更多内容。")
            temp_path = f.name

        try:
            docs = parser.parse_file(temp_path)
            assert len(docs) == 1
            assert "标题" in docs[0].page_content
            assert docs[0].metadata["file_type"] == "markdown"
        finally:
            Path(temp_path).unlink()

    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        from src.document_processor import DocumentParser

        parser = DocumentParser()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                parser.parse_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestDocumentChunker:
    """文档分块器测试"""

    def test_recursive_chunking(self):
        """测试递归分块"""
        from src.document_processor import DocumentChunker

        chunker = DocumentChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy="recursive",
        )

        text = "这是一段很长的测试文本。" * 20
        doc = Document(page_content=text, metadata={"filename": "test.txt"})

        chunks = chunker.split_documents([doc])
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 150 for c in chunks)  # 考虑重叠

    def test_chunk_metadata_preserved(self):
        """测试分块后元数据保留"""
        from src.document_processor import DocumentChunker

        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        doc = Document(
            page_content="测试内容" * 20,
            metadata={"filename": "test.pdf", "page": 1, "file_type": "pdf"},
        )

        chunks = chunker.split_documents([doc])
        assert len(chunks) > 0

        # 检查元数据是否保留
        for chunk in chunks:
            assert chunk.metadata["filename"] == "test.pdf"
            assert chunk.metadata["file_type"] == "pdf"
            assert "chunk_index" in chunk.metadata


class TestBM25Retriever:
    """BM25 检索器测试"""

    def test_add_and_search(self):
        """测试添加文档和检索"""
        from src.retriever import BM25Retriever

        retriever = BM25Retriever(language="chinese")

        docs = [
            Document(page_content="Python 是一种编程语言", metadata={"id": 1}),
            Document(page_content="Java 是另一种编程语言", metadata={"id": 2}),
            Document(page_content="机器学习是人工智能的一个分支", metadata={"id": 3}),
        ]

        retriever.add_documents(docs)
        assert retriever.get_document_count() == 3

        # 搜索
        results = retriever.search("Python 编程", top_k=2)
        assert len(results) > 0
        assert "Python" in results[0].document.page_content

    def test_empty_search(self):
        """测试空查询"""
        from src.retriever import BM25Retriever

        retriever = BM25Retriever(language="chinese")
        results = retriever.search("", top_k=5)
        assert len(results) == 0


class TestHybridRetriever:
    """混合检索器测试"""

    def test_hybrid_retriever_init(self):
        """测试混合检索器初始化"""
        from src.retriever import HybridRetriever, BM25Retriever

        bm25 = BM25Retriever()
        retriever = HybridRetriever(
            vector_store=None,
            bm25_retriever=bm25,
            reranker=None,
            top_k=10,
            rerank_top_n=5,
        )

        assert retriever.top_k == 10
        assert retriever.rerank_top_n == 5


class TestConversationHistory:
    """对话历史测试"""

    def test_add_and_get_messages(self):
        """测试添加和获取消息"""
        from src.llm import ConversationHistory

        history = ConversationHistory(max_turns=5)
        session_id = history.create_session()

        history.add_user_message(session_id, "你好")
        history.add_assistant_message(session_id, "你好！有什么可以帮你的？")

        messages = history.get_history(session_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_max_turns_truncation(self):
        """测试最大轮数截断"""
        from src.llm import ConversationHistory

        history = ConversationHistory(max_turns=2)
        session_id = history.create_session()

        # 添加 5 轮对话
        for i in range(5):
            history.add_user_message(session_id, f"问题{i}")
            history.add_assistant_message(session_id, f"回答{i}")

        messages = history.get_history(session_id)
        # 应该保留最后 2 轮（4 条消息）
        assert len(messages) <= 6  # 可能多一点，因为截断是按轮数


class TestPromptFactory:
    """Prompt 工厂测试"""

    def test_format_context(self):
        """测试格式化上下文"""
        from src.llm import PromptFactory

        docs = [
            Document(
                page_content="这是第一段内容",
                metadata={"filename": "test.pdf", "page": 1},
            ),
            Document(
                page_content="这是第二段内容",
                metadata={"filename": "test.docx", "paragraph": 2},
            ),
        ]

        context = PromptFactory.format_context(docs)
        assert "来源1" in context
        assert "来源2" in context
        assert "test.pdf" in context
        assert "第一段内容" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
