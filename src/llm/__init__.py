"""
LLM 问答模块
LLM Q&A Module

包含：
    - ChatManager: 对话管理器
    - ConversationHistory: 对话历史管理器
    - PromptFactory: Prompt 模板工厂
    - ChatResponse: 对话响应数据类
    - SourceCitation: 引用来源数据类
"""

from src.llm.chat import ChatManager, ChatResponse, SourceCitation
from src.llm.history import ConversationHistory, Message, Conversation
from src.llm.prompt import PromptFactory

__all__ = [
    "ChatManager",
    "ChatResponse",
    "SourceCitation",
    "ConversationHistory",
    "Message",
    "Conversation",
    "PromptFactory",
]
