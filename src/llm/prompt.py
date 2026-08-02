"""
Prompt 模板模块 - 精心设计的问答 Prompt 体系
Prompt Template Module - Well-designed Q&A Prompt System

功能说明：
    提供精心设计的 Prompt 模板，用于引导大模型基于检索到的文档内容
    准确回答用户问题，有效减少幻觉，提升回答质量。

技术选型：
    - LangChain PromptTemplate：结构化 Prompt 管理
    - 多轮对话 Prompt：支持上下文历史
    - 引用标注 Prompt：引导模型标注引用来源

优化策略：
    1. 角色设定：明确模型是"基于文档的问答助手"
    2. 指令约束：严格要求基于文档回答，不知道就说不知道
    3. 引用标注：要求标注引用来源，便于验证
    4. 思维链：引导模型先分析再回答
    5. 格式规范：统一输出格式，便于解析
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


# =============================================================================
# 系统 Prompt - 核心指令
# =============================================================================

SYSTEM_PROMPT = """你是一个专业的企业文档知识库问答助手，你的任务是基于提供的参考文档，准确、专业地回答用户的问题。

## 核心原则
1. **严格基于文档**：所有回答必须完全基于提供的参考文档内容，不得编造、推测或引入文档外的知识。
2. **诚实告知**：如果参考文档中没有相关信息，或者信息不足以回答问题，请明确说"根据现有文档，无法回答这个问题"，绝不能编造答案。
3. **引用标注**：回答中使用的每个关键信息都必须标注引用来源，格式为 [来源N]，其中 N 是参考文档的编号。
4. **准确优先**：宁可回答简短但准确，也不要为了全面性而牺牲准确性。
5. **专业表达**：使用专业、规范的语言，保持客观中立的语气。

## 回答结构
请按照以下结构组织回答：
1. **直接回答**：先用 1-2 句话直接回答问题核心
2. **详细说明**：展开说明具体内容、原因、方法等
3. **引用来源**：在回答末尾列出所有引用的文档来源

## 引用格式
- 正文中引用：在相关内容后标注 [来源N]
- 末尾来源列表：按编号列出每个引用的文档信息

## 注意事项
- 如果文档中存在矛盾的信息，请指出矛盾并分别说明
- 如果问题不够明确，请先说明你的理解再回答
- 不要使用"根据我的知识"、"我认为"等主观表述
- 所有数字、日期、名称等必须与文档完全一致

现在，请基于以下参考文档回答用户的问题：

---
【参考文档开始】
{context}
【参考文档结束】
---"""


# =============================================================================
# 多轮对话 Prompt
# =============================================================================

MULTI_TURN_SYSTEM_PROMPT = """你是一个专业的企业文档知识库问答助手，你的任务是基于提供的参考文档和对话历史，准确、连贯地回答用户的问题。

## 核心原则
1. **严格基于文档**：所有事实性内容必须完全基于提供的参考文档，不得编造。
2. **上下文连贯**：结合对话历史理解用户当前问题，保持对话的连贯性。
3. **诚实告知**：如果文档中没有答案，明确告知用户，绝不编造。
4. **引用标注**：标注信息来源，便于用户验证。

## 对话历史处理
- 对话历史用于理解上下文和用户意图
- 回答事实性问题时，仍以参考文档为准
- 如果用户追问的内容在历史中已回答，可以简要回顾后补充新信息

## 回答结构
1. **直接回答**：简洁明了地回答问题
2. **详细说明**：展开说明具体内容
3. **引用来源**：标注引用的文档来源

---
【参考文档开始】
{context}
【参考文档结束】
---"""


# =============================================================================
# 问题改写 Prompt - 用于多轮对话中的问题优化
# =============================================================================

QUERY_REWRITE_PROMPT = """你是一个问题优化助手。请根据对话历史，将用户的当前问题改写为一个独立、完整、语义明确的问题。

## 任务说明
- 结合对话历史，补全省略的指代和上下文
- 使改写后的问题可以独立被理解，不需要依赖对话历史
- 保持原问题的核心意图，不要添加额外信息
- 如果原问题已经很完整，可以直接返回原问题

## 对话历史
{chat_history}

## 当前问题
{question}

## 改写后的问题（只输出问题本身）："""


# =============================================================================
# 幻觉检测 Prompt
# =============================================================================

HALLUCINATION_CHECK_PROMPT = """请检查以下回答是否完全基于提供的参考文档。

## 参考文档
{context}

## 待检查回答
{answer}

## 检查标准
1. 回答中的所有事实性陈述是否都能在参考文档中找到依据
2. 是否存在文档中没有的数字、日期、名称等具体信息
3. 是否存在过度推断或主观臆断的内容

## 输出格式
请用 JSON 格式输出检查结果：
{{
    "is_hallucination": true/false,
    "confidence": 0-1之间的分数,
    "issues": ["问题1", "问题2", ...],
    "suggestion": "修改建议"
}}

检查结果："""


# =============================================================================
# Prompt 模板工厂
# =============================================================================

class PromptFactory:
    """
    Prompt 模板工厂

    提供各种场景的 Prompt 模板，统一管理，便于维护和优化。
    """

    @staticmethod
    def get_qa_prompt() -> ChatPromptTemplate:
        """
        获取基础问答 Prompt

        Returns:
            ChatPromptTemplate 实例
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="{question}"),
        ])

    @staticmethod
    def get_multi_turn_prompt() -> ChatPromptTemplate:
        """
        获取多轮对话 Prompt

        Returns:
            ChatPromptTemplate 实例
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=MULTI_TURN_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{question}"),
        ])

    @staticmethod
    def get_query_rewrite_prompt() -> PromptTemplate:
        """
        获取问题改写 Prompt

        Returns:
            PromptTemplate 实例
        """
        return PromptTemplate(
            template=QUERY_REWRITE_PROMPT,
            input_variables=["chat_history", "question"],
        )

    @staticmethod
    def get_hallucination_check_prompt() -> PromptTemplate:
        """
        获取幻觉检测 Prompt

        Returns:
            PromptTemplate 实例
        """
        return PromptTemplate(
            template=HALLUCINATION_CHECK_PROMPT,
            input_variables=["context", "answer"],
        )

    @staticmethod
    def format_context(documents: list) -> str:
        """
        格式化参考文档上下文

        Args:
            documents: 文档列表，每个元素应包含 page_content 和 metadata

        Returns:
            格式化后的上下文字符串
        """
        context_parts = []

        for idx, doc in enumerate(documents, start=1):
            # 提取元数据
            metadata = getattr(doc, 'metadata', {})
            filename = metadata.get('filename', '未知文件')
            page = metadata.get('page', '')
            sheet = metadata.get('sheet', '')
            paragraph = metadata.get('paragraph', '')

            # 构建来源信息
            source_info = f"来源{idx}: {filename}"
            if page:
                source_info += f" (第{page}页)"
            if sheet:
                source_info += f" [Sheet: {sheet}]"
            if paragraph:
                source_info += f" (第{paragraph}段)"

            # 文档内容
            content = getattr(doc, 'page_content', str(doc))

            context_parts.append(
                f"【{source_info}】\n{content}\n"
            )

        return "\n".join(context_parts)

    @staticmethod
    def format_chat_history(history: list) -> str:
        """
        格式化对话历史

        Args:
            history: 对话历史列表，每个元素为 (role, content) 元组

        Returns:
            格式化后的对话历史字符串
        """
        if not history:
            return "（无对话历史）"

        parts = []
        for role, content in history:
            role_name = "用户" if role == "user" else "助手"
            parts.append(f"{role_name}: {content}")

        return "\n".join(parts)
