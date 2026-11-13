"""
拒答策略与引用校验的单元测试
================================

这两组逻辑是"抑制幻觉"的核心，且都是纯逻辑（不依赖真实模型与向量库），
因此用假 LLM / 假检索器即可完整覆盖：

- 拒答判定：检索为空 → 拒答；Top1 分数低于阈值 → 拒答；否则正常作答
- 拒答时不调用模型（关键：否则等于把空上下文丢给模型编造）
- 流式分支的拒答行为与非流式一致
- 引用校验：关闭时不产生调用；开启时能解析模型返回的 JSON；
  模型返回垃圾或抛异常时只降级、不打断主流程

运行：pytest tests/test_refusal_and_citation.py -v
"""

from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.llm.chat import ChatManager
from src.retriever.hybrid import RetrievalResult, RetrievalStats


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeLLM:
    """假 LLM：记录调用次数，并按预设返回内容"""

    def __init__(self, content="这是模型的回答 [来源1]", raise_on_invoke=False):
        self.content = content
        self.raise_on_invoke = raise_on_invoke
        self.invoke_calls = 0

    def invoke(self, messages):
        self.invoke_calls += 1
        if self.raise_on_invoke:
            raise RuntimeError("模拟模型接口异常")
        return MagicMock(content=self.content)


class FakeRetriever:
    """假检索器：直接返回给定结果"""

    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, filter=None):
        return self.docs, RetrievalStats(query=query, final_count=len(self.docs))


def make_result(text="参考内容", score=1.0, rank=1):
    return RetrievalResult(
        document=Document(page_content=text, metadata={"filename": "测试文档.txt"}),
        score=score,
        rank=rank,
    )


def build_manager(docs, llm=None, answer_min_score=0.0, enable_citation_check=False):
    return ChatManager(
        llm=llm or FakeLLM(),
        retriever=FakeRetriever(docs),
        enable_query_rewrite=False,
        answer_min_score=answer_min_score,
        enable_citation_check=enable_citation_check,
    )


# ---------------------------------------------------------------------------
# 1. 拒答判定
# ---------------------------------------------------------------------------

def test_should_refuse_when_no_hits():
    manager = build_manager([])
    refuse, reason, score = manager._should_refuse([])
    assert refuse is True
    assert reason == "no_hits"
    assert score == 0.0


def test_should_not_refuse_when_score_above_threshold():
    manager = build_manager([make_result(score=0.9)], answer_min_score=0.5)
    refuse, reason, _ = manager._should_refuse([make_result(score=0.9)])
    assert refuse is False
    assert reason == ""


def test_should_refuse_when_score_below_threshold():
    manager = build_manager([make_result(score=0.2)], answer_min_score=0.5)
    refuse, reason, score = manager._should_refuse([make_result(score=0.2)])
    assert refuse is True
    assert reason == "low_score"
    assert score == pytest.approx(0.2)


def test_default_threshold_only_refuses_empty_result():
    """默认 answer_min_score=0.0 时，只要检索到内容就不拒答（不改变既有行为）"""
    manager = build_manager([make_result(score=0.01)])
    refuse, _, _ = manager._should_refuse([make_result(score=0.01)])
    assert refuse is False


def test_refusal_answer_contains_actionable_suggestion():
    manager = build_manager([])
    text = manager._build_refusal_answer("no_hits", 0.0)
    assert "未在知识库中检索到" in text
    assert "建议尝试" in text


# ---------------------------------------------------------------------------
# 2. 拒答时不调用模型（防编造的关键断言）
# ---------------------------------------------------------------------------

def test_chat_refuses_without_calling_llm():
    llm = FakeLLM()
    manager = build_manager([], llm=llm)

    response = manager.chat("知识库里没有的主题是什么？")

    assert response.metadata["refused"] is True
    assert response.metadata["refuse_reason"] == "no_hits"
    assert response.sources == []
    assert llm.invoke_calls == 0, "拒答路径不应调用大模型"
    assert "未在知识库中检索到" in response.answer


def test_chat_normal_path_records_citation_check_disabled():
    llm = FakeLLM()
    manager = build_manager([make_result()], llm=llm)

    response = manager.chat("有材料支撑的问题")

    assert response.metadata.get("refused") is not True
    assert response.metadata["citation_check"] == {"enabled": False}
    assert llm.invoke_calls >= 1


def test_chat_stream_refuses_consistently():
    manager = build_manager([], llm=FakeLLM())

    chunks = []
    final = None
    for piece, resp in manager.chat_stream("没有材料的问题"):
        if piece:
            chunks.append(piece)
        if resp is not None:
            final = resp

    assert final is not None
    assert final.metadata["refused"] is True
    assert "".join(chunks).replace("", "") != ""
    assert "未在知识库中检索到" in final.answer


# ---------------------------------------------------------------------------
# 3. 引用/幻觉校验
# ---------------------------------------------------------------------------

def test_citation_check_parses_valid_json():
    llm = FakeLLM(content='{"is_hallucination": false, "confidence": 0.92, "issues": [], "suggestion": "无需修改"}')
    manager = build_manager([make_result()], llm=llm, enable_citation_check=True)

    result = manager._verify_answer("问题", "上下文", "回答")

    assert result["enabled"] is True
    assert result["is_hallucination"] is False
    assert result["confidence"] == pytest.approx(0.92)


def test_citation_check_handles_code_fence_wrapped_json():
    llm = FakeLLM(content='```json\n{"is_hallucination": true, "issues": ["第2句无依据"]}\n```')
    manager = build_manager([make_result()], llm=llm, enable_citation_check=True)

    result = manager._verify_answer("问题", "上下文", "回答")

    assert result["is_hallucination"] is True
    assert result["issues"] == ["第2句无依据"]


def test_citation_check_degrades_on_invalid_json():
    llm = FakeLLM(content="我觉得这个回答没问题")
    manager = build_manager([make_result()], llm=llm, enable_citation_check=True)

    result = manager._verify_answer("问题", "上下文", "回答")

    assert result["enabled"] is True
    assert result["parse_error"] == "模型未返回合法 JSON"
    assert "raw" in result


def test_citation_check_never_raises_when_llm_fails():
    llm = FakeLLM(raise_on_invoke=True)
    manager = build_manager([make_result()], llm=llm, enable_citation_check=True)

    result = manager._verify_answer("问题", "上下文", "回答")

    assert result["enabled"] is True
    assert "error" in result


def test_citation_check_skipped_when_disabled():
    llm = FakeLLM()
    manager = build_manager([make_result()], llm=llm, enable_citation_check=False)

    assert manager._verify_answer("问题", "上下文", "回答") == {"enabled": False}
    assert llm.invoke_calls == 0
