"""
重排器单元测试
================

覆盖两件事：
    1. 词法重排器（LexicalReranker）的打分与排序行为是确定性的、可解释的；
    2. 神经重排器在依赖/模型缺失时的降级行为是"保持原顺序"，
       而不是一个会把排序搞坏却又声称"已重排"的假实现。

背景：`scripts/ab_rerank.py` 的对照实验显示，词法重排在已饱和的召回上
反而降低 Recall@1（256 分块 97.5% → 85.0%），所以它只作为**评测基线**
与独立组件保留，没有接进生产降级路径——这里的用例正是把这条边界钉死。
"""

import pytest
from langchain_core.documents import Document

from src.retriever.lexical_reranker import LexicalReranker, _tokenize
from src.retriever.reranker import BGEReranker


def make_doc(text, name="doc.txt"):
    return Document(page_content=text, metadata={"filename": name})


# ---------------------------------------------------------------------------
# 1. 词法重排器
# ---------------------------------------------------------------------------

def test_rerank_empty_documents_returns_empty():
    assert LexicalReranker().rerank("任意问题", [], top_n=5) == []


def test_rerank_promotes_document_containing_query_terms():
    docs = [
        make_doc("这段内容讲的是数据清洗与去重流程。", "a.txt"),
        make_doc("向量维度必须与嵌入模型保持一致，否则相似度计算失真。", "b.txt"),
        make_doc("报告导出支持 Markdown 与 PDF 两种格式。", "c.txt"),
    ]
    results = LexicalReranker().rerank("向量维度和嵌入模型不一致会怎样", docs, top_n=3)

    assert results[0].document.metadata["filename"] == "b.txt"
    assert results[0].new_rank == 1
    # 原始排名必须保留，便于排查"重排把谁提上来了"
    assert results[0].original_rank == 2


def test_rerank_truncates_to_top_n():
    docs = [make_doc(f"文档内容 {i} 主题测试", f"{i}.txt") for i in range(6)]
    results = LexicalReranker().rerank("主题测试", docs, top_n=3)
    assert len(results) == 3
    assert [r.new_rank for r in results] == [1, 2, 3]


def test_rerank_is_deterministic():
    docs = [make_doc("内容甲 RAG 检索", "a.txt"), make_doc("内容乙 RAG 检索", "b.txt")]
    first = [r.original_rank for r in LexicalReranker().rerank("RAG 检索", docs, top_n=2)]
    second = [r.original_rank for r in LexicalReranker().rerank("RAG 检索", docs, top_n=2)]
    assert first == second


def test_score_is_zero_when_no_term_overlap():
    assert LexicalReranker().score("完全无关的问题", "另一段毫不相干的文字") == 0.0


def test_tokenize_drops_stopwords_and_single_chars():
    tokens = _tokenize("这个东西的检索是如何做的")
    assert "的" not in tokens
    assert all(len(t) > 1 for t in tokens)


# ---------------------------------------------------------------------------
# 2. 神经重排器的降级行为
# ---------------------------------------------------------------------------

def test_bge_reranker_degrades_to_original_order():
    """FlagEmbedding 未安装时，必须保持原序（而不是用弱基线乱排）"""
    reranker = BGEReranker(model_name="BAAI/bge-reranker-large")
    docs = [
        make_doc("第一段：完全不提问题里的词。"),
        make_doc("第二段：包含 向量 与 维度 两个词。"),
        make_doc("第三段：随便写点别的。"),
    ]

    results = reranker.rerank("向量维度", docs, top_n=3)

    assert [r.original_rank for r in results] == [1, 2, 3]
    assert [r.new_rank for r in results] == [1, 2, 3]
    assert results[0].document.page_content.startswith("第一段")
