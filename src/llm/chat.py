"""
对话管理模块 - 多轮对话问答核心
Chat Manager Module - Multi-turn Q&A Core

功能说明：
    实现基于 RAG 的问答对话系统，整合检索、Prompt 构建、LLM 调用、
    流式输出、引用标注等功能，提供完整的问答体验。

技术选型：
    - LangChain Chat Models：统一的 LLM 接口
    - 流式输出：支持 token-by-token 生成
    - 引用解析：自动提取和标注引用来源
    - 问题改写：优化多轮对话中的问题理解

优化策略：
    1. 问题改写：结合历史上下文改写问题，提升检索准确率
    2. 流式输出：实时返回生成内容，提升用户体验
    3. 引用标注：自动提取和格式化引用来源
    4. 幻觉抑制：严格的 Prompt 约束 + 引用验证
    5. 错误处理：优雅处理各种异常情况
"""

from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from src.llm.prompt import PromptFactory
from src.llm.history import ConversationHistory
from src.retriever.hybrid import HybridRetriever, RetrievalResult


@dataclass
class SourceCitation:
    """引用来源"""
    index: int
    filename: str
    page: Optional[str] = None
    sheet: Optional[str] = None
    paragraph: Optional[str] = None
    score: float = 0.0
    snippet: str = ""


@dataclass
class ChatResponse:
    """对话响应"""
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)
    retrieved_docs: List[RetrievalResult] = field(default_factory=list)
    session_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatManager:
    """
    对话管理器

    核心功能：
        - 基于 RAG 的问答
        - 多轮对话支持
        - 流式输出
        - 引用来源标注
        - 问题改写优化
    """

    def __init__(
        self,
        llm,
        retriever: HybridRetriever,
        history: Optional[ConversationHistory] = None,
        enable_streaming: bool = True,
        enable_query_rewrite: bool = True,
        max_history_turns: int = 10,
        answer_min_score: float = 0.0,
        enable_citation_check: bool = False,
    ):
        """
        初始化对话管理器

        Args:
            llm: LLM 实例
            retriever: 混合检索器
            history: 对话历史管理器（可选）
            enable_streaming: 是否启用流式输出
            enable_query_rewrite: 是否启用问题改写
            max_history_turns: 最大历史轮数
            answer_min_score: 拒答阈值。Top1 检索分数低于该值时直接拒答，
                不调用大模型（默认 0.0 表示只对"完全没检索到内容"拒答）
            enable_citation_check: 是否在生成后做一次幻觉/引用校验（多一次模型调用）
        """
        self.llm = llm
        self.retriever = retriever
        self.history = history or ConversationHistory(max_turns=max_history_turns)
        self.enable_streaming = enable_streaming
        self.enable_query_rewrite = enable_query_rewrite
        self.max_history_turns = max_history_turns
        self.answer_min_score = answer_min_score
        self.enable_citation_check = enable_citation_check

        # Prompt 工厂
        self.prompt_factory = PromptFactory()

        logger.info(
            f"对话管理器初始化完成: "
            f"流式输出={enable_streaming}, "
            f"问题改写={enable_query_rewrite}, "
            f"最大历史轮数={max_history_turns}, "
            f"拒答阈值={answer_min_score}, "
            f"引用校验={enable_citation_check}"
        )

    def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        非流式问答

        Args:
            question: 用户问题
            session_id: 会话 ID（可选，新会话自动创建）
            filter: 检索过滤条件

        Returns:
            ChatResponse 响应对象
        """
        import time
        start_time = time.time()

        # 创建或获取会话
        if session_id is None:
            session_id = self.history.create_session()

        try:
            # 1. 问题改写（多轮对话）
            optimized_question = self._rewrite_query(question, session_id)

            # 2. 检索相关文档
            retrieved_docs, stats = self.retriever.retrieve(
                query=optimized_question,
                filter=filter,
            )

            # 2.5 拒答判定：检索为空或 Top1 分数低于阈值时不调用模型，
            #     直接给出"未找到"回答，避免空上下文下模型自由编造
            refuse, refuse_reason, top1_score = self._should_refuse(retrieved_docs)
            if refuse:
                refusal = self._build_refusal_answer(refuse_reason, top1_score)
                self.history.add_user_message(session_id, question)
                self.history.add_assistant_message(session_id, refusal, {
                    "refused": True,
                    "refuse_reason": refuse_reason,
                    "retrieval_stats": stats.__dict__,
                })
                return ChatResponse(
                    answer=refusal,
                    sources=[],
                    retrieved_docs=retrieved_docs,
                    session_id=session_id,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "retrieval_stats": stats.__dict__,
                        "optimized_question": optimized_question,
                        "refused": True,
                        "refuse_reason": refuse_reason,
                        "top1_score": top1_score,
                    },
                )

            # 3. 构建 Prompt
            context = self.prompt_factory.format_context(
                [r.document for r in retrieved_docs]
            )

            # 4. 调用 LLM
            answer = self._generate_answer(
                question=optimized_question,
                context=context,
                session_id=session_id,
            )

            # 5. 解析引用来源
            sources = self._extract_sources(answer, retrieved_docs)

            # 5.5 引用校验（可选）：检查答案是否被检索材料支持
            citation_check = self._verify_answer(question, context, answer)

            # 6. 保存对话历史
            self.history.add_user_message(session_id, question)
            self.history.add_assistant_message(session_id, answer, {
                "sources": [s.__dict__ for s in sources],
                "retrieval_stats": stats.__dict__,
            })

            # 计算耗时
            latency = (time.time() - start_time) * 1000

            return ChatResponse(
                answer=answer,
                sources=sources,
                retrieved_docs=retrieved_docs,
                session_id=session_id,
                latency_ms=latency,
                metadata={
                    "retrieval_stats": stats.__dict__,
                    "optimized_question": optimized_question,
                    "citation_check": citation_check,
                },
            )

        except Exception as e:
            logger.error(f"问答失败: {str(e)}")
            raise

    def chat_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[str, Optional[ChatResponse]], None, None]:
        """
        流式问答

        Args:
            question: 用户问题
            session_id: 会话 ID
            filter: 检索过滤条件

        Yields:
            (内容片段, 最终响应（仅最后一次有值）)
        """
        import time
        start_time = time.time()

        # 创建或获取会话
        if session_id is None:
            session_id = self.history.create_session()

        try:
            # 1. 问题改写
            optimized_question = self._rewrite_query(question, session_id)

            # 2. 检索相关文档
            retrieved_docs, stats = self.retriever.retrieve(
                query=optimized_question,
                filter=filter,
            )

            # 2.5 拒答判定（与非流式保持一致的口径）
            refuse, refuse_reason, top1_score = self._should_refuse(retrieved_docs)
            if refuse:
                refusal = self._build_refusal_answer(refuse_reason, top1_score)
                self.history.add_user_message(session_id, question)
                self.history.add_assistant_message(session_id, refusal, {
                    "refused": True,
                    "refuse_reason": refuse_reason,
                    "retrieval_stats": stats.__dict__,
                })
                for piece in self._chunk_text(refusal):
                    yield piece, None
                yield "", ChatResponse(
                    answer=refusal,
                    sources=[],
                    retrieved_docs=retrieved_docs,
                    session_id=session_id,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "retrieval_stats": stats.__dict__,
                        "optimized_question": optimized_question,
                        "refused": True,
                        "refuse_reason": refuse_reason,
                        "top1_score": top1_score,
                    },
                )
                return

            # 3. 构建 Prompt
            context = self.prompt_factory.format_context(
                [r.document for r in retrieved_docs]
            )

            # 4. 流式生成
            full_answer = ""
            for chunk in self._stream_answer(
                question=optimized_question,
                context=context,
                session_id=session_id,
            ):
                full_answer += chunk
                yield chunk, None

            # 5. 解析引用来源
            sources = self._extract_sources(full_answer, retrieved_docs)

            # 5.5 引用校验（可选）
            citation_check = self._verify_answer(question, context, full_answer)

            # 6. 保存对话历史
            self.history.add_user_message(session_id, question)
            self.history.add_assistant_message(session_id, full_answer, {
                "sources": [s.__dict__ for s in sources],
                "retrieval_stats": stats.__dict__,
            })

            # 计算耗时
            latency = (time.time() - start_time) * 1000

            # 返回最终响应
            final_response = ChatResponse(
                answer=full_answer,
                sources=sources,
                retrieved_docs=retrieved_docs,
                session_id=session_id,
                latency_ms=latency,
                metadata={
                    "retrieval_stats": stats.__dict__,
                    "optimized_question": optimized_question,
                    "citation_check": citation_check,
                },
            )

            yield "", final_response

        except Exception as e:
            logger.error(f"流式问答失败: {str(e)}")
            raise

    def _rewrite_query(self, question: str, session_id: str) -> str:
        """
        改写问题（结合对话历史）

        Args:
            question: 原始问题
            session_id: 会话 ID

        Returns:
            改写后的问题
        """
        if not self.enable_query_rewrite:
            return question

        # 获取历史消息
        history = self.history.get_history_for_prompt(session_id)

        # 如果是第一轮对话，不需要改写
        if not history:
            return question

        try:
            # 构建改写 Prompt
            rewrite_prompt = self.prompt_factory.get_query_rewrite_prompt()
            chat_history_str = self.prompt_factory.format_chat_history(history)

            # 调用 LLM 改写
            messages = [
                HumanMessage(content=rewrite_prompt.format(
                    chat_history=chat_history_str,
                    question=question,
                ))
            ]

            response = self.llm.invoke(messages)
            rewritten = response.content.strip()

            logger.debug(f"问题改写: '{question}' -> '{rewritten}'")
            return rewritten if rewritten else question

        except Exception as e:
            logger.warning(f"问题改写失败: {str(e)}，使用原始问题")
            return question

    def _generate_answer(
        self,
        question: str,
        context: str,
        session_id: str,
    ) -> str:
        """
        生成回答（非流式）

        Args:
            question: 问题
            context: 上下文
            session_id: 会话 ID

        Returns:
            回答文本
        """
        # 获取对话历史
        history_messages = self._get_history_messages(session_id)

        # 构建 Prompt
        if history_messages:
            # 多轮对话
            prompt = self.prompt_factory.get_multi_turn_prompt()
            messages = prompt.format_messages(
                context=context,
                question=question,
                chat_history=history_messages,
            )
        else:
            # 单轮对话
            prompt = self.prompt_factory.get_qa_prompt()
            messages = prompt.format_messages(
                context=context,
                question=question,
            )

        # 调用 LLM
        response = self.llm.invoke(messages)
        return response.content

    def _should_refuse(self, retrieved_docs: List[RetrievalResult]) -> Tuple[bool, str, float]:
        """
        判断是否应当拒答

        判定口径（两条，先到先触发）：
            1. 检索结果为空（no_hits）—— 没有任何材料支撑，模型只能编；
            2. Top1 检索分数低于 answer_min_score（low_score）—— 有结果但不够相关。

        注意：默认 answer_min_score=0.0，此时只有"检索完全为空"才拒答，
        因此不改变既有行为的语义，只是把"空上下文下编造"这条路径堵住。

        Returns:
            (是否拒答, 拒答原因, Top1 分数)
        """
        if not retrieved_docs:
            return True, "no_hits", 0.0

        top1_score = float(getattr(retrieved_docs[0], "score", 0.0) or 0.0)
        if top1_score < self.answer_min_score:
            return True, "low_score", top1_score

        return False, "", top1_score

    def _build_refusal_answer(self, reason: str, top1_score: float) -> str:
        """构造拒答文案：明确说明未找到，并给出可执行的下一步"""
        if reason == "low_score":
            head = (
                f"未在知识库中找到足够相关的内容（最高相关度 {top1_score:.2f}，"
                f"低于当前阈值 {self.answer_min_score:.2f}），因此不作答以免给出错误信息。"
            )
        else:
            head = "未在知识库中检索到与该问题相关的内容，因此不作答以免给出错误信息。"
        return (
            f"{head}\n\n"
            "建议尝试：\n"
            "1. 换一种说法或补充关键信息（文档名、章节、具体术语）；\n"
            "2. 确认相关文档已经上传并完成索引；\n"
            "3. 如果是新主题，先补充材料再提问。"
        )

    def _verify_answer(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        引用/幻觉校验：检查回答是否被检索到的材料支持

        设计要点：
            - 默认关闭（enable_citation_check=False），因为每次问答会多一次模型调用；
            - 结果只放进 metadata 供上层展示，不擅自修改回答内容；
            - 任何异常都降级为 {"enabled": True, "error": ...}，绝不打断主流程。

        Returns:
            校验结果字典
        """
        if not self.enable_citation_check:
            return {"enabled": False}

        try:
            prompt = self.prompt_factory.get_hallucination_check_prompt()
            # 该校验用的是 PromptTemplate（字符串模板），不是 ChatPromptTemplate
            prompt_text = prompt.format(context=context, answer=answer)
            messages = [HumanMessage(content=prompt_text)]
            raw = self.llm.invoke(messages).content or ""

            # 模型可能返回带代码块围栏的 JSON，先剥掉再解析
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            import json
            try:
                parsed = json.loads(cleaned)
                return {
                    "enabled": True,
                    "is_hallucination": bool(parsed.get("is_hallucination", False)),
                    "confidence": parsed.get("confidence"),
                    "issues": parsed.get("issues", []),
                    "suggestion": parsed.get("suggestion", ""),
                }
            except json.JSONDecodeError:
                # 解析失败不算失败，保留原文便于人工查看
                return {"enabled": True, "raw": raw[:500], "parse_error": "模型未返回合法 JSON"}
        except Exception as e:
            logger.warning(f"引用校验失败（不影响回答）: {str(e)}")
            return {"enabled": True, "error": str(e)}

    @staticmethod
    def _chunk_text(text: str, size: int = 12) -> Generator[str, None, None]:
        """把拒答文案切成小片，让流式接口在拒答时也有输出节奏"""
        for i in range(0, len(text), size):
            yield text[i : i + size]

    def _stream_answer(
        self,
        question: str,
        context: str,
        session_id: str,
    ) -> Generator[str, None, None]:
        """
        流式生成回答

        Args:
            question: 问题
            context: 上下文
            session_id: 会话 ID

        Yields:
            文本片段
        """
        # 获取对话历史
        history_messages = self._get_history_messages(session_id)

        # 构建 Prompt
        if history_messages:
            prompt = self.prompt_factory.get_multi_turn_prompt()
            messages = prompt.format_messages(
                context=context,
                question=question,
                chat_history=history_messages,
            )
        else:
            prompt = self.prompt_factory.get_qa_prompt()
            messages = prompt.format_messages(
                context=context,
                question=question,
            )

        # 流式调用
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def _get_history_messages(self, session_id: str) -> List[BaseMessage]:
        """
        获取历史消息（LangChain 格式）

        Args:
            session_id: 会话 ID

        Returns:
            消息列表
        """
        history = self.history.get_history(session_id)

        messages = []
        for msg in history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        return messages

    def _extract_sources(
        self,
        answer: str,
        retrieved_docs: List[RetrievalResult],
    ) -> List[SourceCitation]:
        """
        从回答中提取引用来源

        Args:
            answer: 回答文本
            retrieved_docs: 检索到的文档

        Returns:
            引用来源列表
        """
        import re

        sources = []
        cited_indices = set()

        # 匹配 [来源N] 格式
        pattern = r'\[来源(\d+)\]'
        matches = re.findall(pattern, answer)

        for match in matches:
            try:
                idx = int(match)
                if 1 <= idx <= len(retrieved_docs):
                    cited_indices.add(idx - 1)  # 转换为 0-based
            except ValueError:
                continue

        # 如果没有明确引用，返回所有检索到的文档
        if not cited_indices:
            cited_indices = set(range(min(len(retrieved_docs), 5)))

        # 构建引用来源列表
        for idx in sorted(cited_indices):
            if idx < len(retrieved_docs):
                doc_result = retrieved_docs[idx]
                doc = doc_result.document
                meta = doc.metadata

                # 提取片段（前 100 字符）
                snippet = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content

                source = SourceCitation(
                    index=idx + 1,
                    filename=meta.get("filename", "未知文件"),
                    page=meta.get("page"),
                    sheet=meta.get("sheet"),
                    paragraph=meta.get("paragraph"),
                    score=doc_result.score,
                    snippet=snippet,
                )
                sources.append(source)

        return sources

    def clear_history(self, session_id: str) -> None:
        """
        清空会话历史

        Args:
            session_id: 会话 ID
        """
        self.history.clear_history(session_id)
        logger.info(f"清空会话历史: {session_id}")

    def clear_all_sessions(self) -> int:
        """
        清空全部对话历史（删除所有会话）

        Returns:
            被清空的会话数量
        """
        count = self.history.clear_all_sessions()
        logger.info(f"已清空全部对话历史，共 {count} 个会话")
        return count

    def get_history(self, session_id: str) -> list:
        """
        获取会话历史

        Args:
            session_id: 会话 ID

        Returns:
            消息列表
        """
        return self.history.get_history(session_id)

    def list_sessions(self) -> List[Dict]:
        """列出所有会话"""
        return self.history.list_sessions()
