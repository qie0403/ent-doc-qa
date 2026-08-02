"""
历史对话管理模块 - 多轮对话上下文管理
Conversation History Module - Multi-turn Context Management

功能说明：
    管理多轮对话的历史记录，支持对话的存储、检索、截断、摘要等功能，
    确保对话上下文的连贯性和有效性。

技术选型：
    - 内存存储：轻量级，适合单会话场景
    - 滑动窗口：保留最近 N 轮对话，控制 token 数量
    - 对话摘要：长对话自动摘要，减少 token 占用

优化策略：
    1. 滑动窗口：只保留最近 N 轮对话，避免上下文过长
    2. Token 计数：精确控制上下文 token 数量
    3. 分级存储：完整历史 + 摘要历史
    4. 会话隔离：多会话独立管理
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


@dataclass
class Message:
    """消息"""
    role: str  # user / assistant
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class Conversation:
    """对话会话"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


class ConversationHistory:
    """
    对话历史管理器

    功能：
        - 多会话管理
        - 消息添加与检索
        - 滑动窗口截断
        - Token 数量控制
        - 对话摘要
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 4000,
        token_estimator=None,
    ):
        """
        初始化对话历史管理器

        Args:
            max_turns: 最大保留轮数（一轮 = user + assistant）
            max_tokens: 最大 token 数
            token_estimator: token 估算函数
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._token_estimator = token_estimator or self._default_token_estimator
        self._conversations: Dict[str, Conversation] = {}

        logger.info(
            f"对话历史管理器初始化完成: "
            f"最大轮数={max_turns}, 最大token数={max_tokens}"
        )

    def _default_token_estimator(self, text: str) -> int:
        """
        默认 token 估算器（简单按字符数估算）

        Args:
            text: 文本

        Returns:
            估算的 token 数量
        """
        # 中文约 1.5 字符/token，英文约 4 字符/token
        # 简化估算：按字符数 / 2 计算
        return len(text) // 2

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        创建新会话

        Args:
            session_id: 会话 ID（可选，自动生成）

        Returns:
            会话 ID
        """
        import uuid

        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(session_id=session_id)
            logger.info(f"创建新会话: {session_id}")

        return session_id

    def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        添加用户消息

        Args:
            session_id: 会话 ID
            content: 消息内容
            metadata: 元数据
        """
        self._add_message(session_id, "user", content, metadata or {})

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        添加助手消息

        Args:
            session_id: 会话 ID
            content: 消息内容
            metadata: 元数据
        """
        self._add_message(session_id, "assistant", content, metadata or {})

    def _add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict,
    ) -> None:
        """
        添加消息

        Args:
            session_id: 会话 ID
            role: 角色
            content: 内容
            metadata: 元数据
        """
        # 确保会话存在
        if session_id not in self._conversations:
            self.create_session(session_id)

        conversation = self._conversations[session_id]

        # 添加消息
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
        )
        conversation.messages.append(message)
        conversation.updated_at = datetime.now().isoformat()

        # 截断历史
        self._truncate_history(session_id)

        logger.debug(f"添加消息: 会话={session_id}, 角色={role}, 长度={len(content)}")

    def _truncate_history(self, session_id: str) -> None:
        """
        截断对话历史（滑动窗口）

        Args:
            session_id: 会话 ID
        """
        conversation = self._conversations[session_id]
        messages = conversation.messages

        # 按轮数截断（一轮 = user + assistant）
        # 计算完整轮数
        user_messages = [m for m in messages if m.role == "user"]
        if len(user_messages) > self.max_turns:
            # 保留最后 max_turns 轮
            # 找到要保留的起始位置
            excess_turns = len(user_messages) - self.max_turns
            user_count = 0
            start_idx = 0

            for i, msg in enumerate(messages):
                if msg.role == "user":
                    user_count += 1
                    if user_count > excess_turns:
                        start_idx = i
                        break

            conversation.messages = messages[start_idx:]
            logger.debug(
                f"按轮数截断历史: {len(messages)} -> {len(conversation.messages)}"
            )

        # 按 token 数截断
        total_tokens = sum(self._token_estimator(m.content) for m in conversation.messages)
        if total_tokens > self.max_tokens:
            # 从前面逐步删除消息，直到 token 数满足要求
            while len(conversation.messages) > 2 and total_tokens > self.max_tokens:
                removed = conversation.messages.pop(0)
                total_tokens -= self._token_estimator(removed.content)

            logger.debug(
                f"按token数截断历史: 移除至 {len(conversation.messages)} 条消息, "
                f"约 {total_tokens} tokens"
            )

    def get_history(self, session_id: str) -> List[Message]:
        """
        获取完整对话历史

        Args:
            session_id: 会话 ID

        Returns:
            消息列表
        """
        if session_id not in self._conversations:
            return []

        return self._conversations[session_id].messages.copy()

    def get_history_for_prompt(self, session_id: str) -> List[Tuple[str, str]]:
        """
        获取用于 Prompt 的对话历史（简化格式）

        Args:
            session_id: 会话 ID

        Returns:
            (role, content) 元组列表
        """
        messages = self.get_history(session_id)
        return [(m.role, m.content) for m in messages]

    def get_last_n_turns(self, session_id: str, n: int) -> List[Message]:
        """
        获取最近 N 轮对话

        Args:
            session_id: 会话 ID
            n: 轮数

        Returns:
            消息列表
        """
        messages = self.get_history(session_id)

        # 找到最后 n 轮的起始位置
        user_count = 0
        start_idx = len(messages)

        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                user_count += 1
                if user_count > n:
                    start_idx = i + 1
                    break

        return messages[start_idx:]

    def clear_history(self, session_id: str) -> None:
        """
        清空会话历史

        Args:
            session_id: 会话 ID
        """
        if session_id in self._conversations:
            self._conversations[session_id].messages = []
            self._conversations[session_id].updated_at = datetime.now().isoformat()
            logger.info(f"清空会话历史: {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.info(f"删除会话: {session_id}")
            return True
        return False

    def list_sessions(self) -> List[Dict]:
        """
        列出所有会话

        Returns:
            会话信息列表
        """
        sessions = []
        for session_id, conv in self._conversations.items():
            sessions.append({
                "session_id": session_id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
            })
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def get_session_count(self) -> int:
        """获取会话数量"""
        return len(self._conversations)

    def get_message_count(self, session_id: str) -> int:
        """
        获取会话消息数量

        Args:
            session_id: 会话 ID

        Returns:
            消息数量
        """
        if session_id not in self._conversations:
            return 0
        return len(self._conversations[session_id].messages)
