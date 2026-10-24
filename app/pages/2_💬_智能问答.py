"""
智能问答页面 - 多轮对话、引用标注、流式输出
Intelligent Q&A Page - Multi-turn Conversation, Citations, Streaming

功能：
    - 多轮对话问答
    - 流式输出
    - 引用来源标注
    - 历史对话管理
    - 会话切换
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

# 页面配置
st.set_page_config(
    page_title="智能问答 - DocMind 智析",
    page_icon="💬",
    layout="wide",
)

# 自定义 CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        max-width: 85%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background: #f0f2f5;
        color: #333;
        margin-right: auto;
    }
    .source-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        font-size: 0.85rem;
    }
    .source-title {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.25rem;
    }
    .source-meta {
        color: #666;
        font-size: 0.8rem;
    }
    .source-snippet {
        color: #555;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .session-item {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background 0.2s;
    }
    .session-item:hover {
        background: #f0f2f5;
    }
    .session-item.active {
        background: #e3effb;
        border-left: 3px solid #0d47a1;
    }
</style>
""", unsafe_allow_html=True)


def check_initialization():
    """检查系统是否已初始化"""
    if "pipeline" not in st.session_state or not st.session_state.pipeline:
        st.error("❌ 系统未初始化，请先回到首页初始化系统")
        if st.button("🏠 返回首页"):
            st.switch_page("pages/home.py")
        st.stop()

    return st.session_state.pipeline


def init_chat_state():
    """初始化聊天状态"""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}  # session_id -> messages


def get_current_session_id():
    """获取当前会话 ID"""
    if st.session_state.current_session_id is None:
        # 创建新会话
        pipeline = st.session_state.pipeline
        session_id = pipeline.conversation_history.create_session()
        st.session_state.current_session_id = session_id
        st.session_state.chat_history[session_id] = []

    return st.session_state.current_session_id


def display_message(role, content, sources=None):
    """
    显示聊天消息

    Args:
        role: 角色（user/assistant）
        content: 消息内容
        sources: 引用来源列表
    """
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-message user-message">
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-message assistant-message">
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 显示引用来源
        if sources:
            with st.expander(f"📚 引用来源 ({len(sources)} 个)"):
                for source in sources:
                    source_info = source.get("filename", "未知文件")
                    if source.get("page"):
                        source_info += f" · 第{source['page']}页"
                    if source.get("sheet"):
                        source_info += f" · Sheet: {source['sheet']}"

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-title">[来源{source.get('index', '?')}] {source_info}</div>
                            <div class="source-meta">相关度: {source.get('score', 0):.2%}</div>
                            <div class="source-snippet">"{source.get('snippet', '')}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def main():
    """智能问答页面主函数"""
    pipeline = check_initialization()
    init_chat_state()

    # 页面标题
    st.title("💬 智能问答")
    st.markdown("基于您的文档知识库，智能回答您的问题")

    # 侧边栏
    with st.sidebar:
        st.subheader("💬 对话会话")

        # 新建会话按钮
        if st.button("➕ 新建会话", type="primary", width='stretch'):
            session_id = pipeline.conversation_history.create_session()
            st.session_state.current_session_id = session_id
            st.session_state.chat_history[session_id] = []
            st.rerun()

        st.markdown("---")

        # 会话列表
        sessions = pipeline.list_sessions()

        if not sessions:
            st.info("暂无会话")
        else:
            current_id = get_current_session_id()

            for session in sessions:
                session_id = session["session_id"]
                msg_count = session["message_count"]

                # 显示会话名称（使用第一条消息作为标题）
                session_title = f"会话 ({msg_count} 条消息)"
                if session_id in st.session_state.chat_history and st.session_state.chat_history[session_id]:
                    first_msg = st.session_state.chat_history[session_id][0]
                    if first_msg["role"] == "user":
                        session_title = first_msg["content"][:20] + "..."

                is_active = session_id == current_id
                css_class = "session-item active" if is_active else "session-item"

                if st.button(
                    session_title,
                    key=f"session_{session_id}",
                    width='stretch',
                    type="secondary",
                ):
                    st.session_state.current_session_id = session_id
                    st.rerun()

        st.markdown("---")

        # 清空当前会话
        if st.button("🗑️ 清空当前会话", width='stretch'):
            current_id = get_current_session_id()
            pipeline.clear_session(current_id)
            st.session_state.chat_history[current_id] = []
            st.rerun()

        # 清空全部对话历史（二次确认，防止误删）
        if st.button("🧹 清空全部对话历史", width='stretch'):
            st.session_state.confirm_clear_all = True

        if st.session_state.get("confirm_clear_all"):
            st.warning("⚠️ 将删除所有会话记录，此操作不可恢复，确定吗？")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("✅ 确定清空", key="confirm_clear_all_btn", type="primary"):
                    cleared_count = pipeline.clear_all_sessions()
                    st.session_state.chat_history = {}
                    st.session_state.current_session_id = None
                    st.session_state.confirm_clear_all = False
                    st.toast(f"已清空全部对话历史（{cleared_count} 个会话）", icon="🧹")
                    st.rerun()
            with cancel_col:
                if st.button("❌ 取消", key="cancel_clear_all_btn"):
                    st.session_state.confirm_clear_all = False
                    st.rerun()

        st.markdown("---")

        if st.button("🏠 返回首页", width='stretch'):
            st.switch_page("pages/home.py")

    # 主聊天区域
    session_id = get_current_session_id()

    # 聊天消息容器
    chat_container = st.container()

    with chat_container:
        # 显示历史消息
        if session_id in st.session_state.chat_history:
            for msg in st.session_state.chat_history[session_id]:
                display_message(
                    role=msg["role"],
                    content=msg["content"],
                    sources=msg.get("sources"),
                )
        else:
            # 欢迎消息
            st.markdown(
                """
                <div class="chat-message assistant-message">
                    👋 您好！我是您的文档知识库助手。<br><br>
                    我可以基于您上传的文档回答问题，支持多轮对话、引用来源标注。<br><br>
                    请在下方输入您的问题，我会尽力为您解答！
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 输入区域
    st.markdown("---")

    # 快捷问题
    st.markdown("**💡 快捷问题**")
    quick_questions = [
        "知识库中有哪些类型的文档？",
        "请总结一下主要内容",
        "有哪些重要的概念？",
    ]

    cols = st.columns(len(quick_questions))
    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(q, width='stretch', key=f"quick_{i}"):
                st.session_state.pending_question = q
                st.rerun()

    # 文本输入
    user_input = st.chat_input("输入您的问题...", key="chat_input")

    # 处理待处理的问题（快捷问题或输入的问题）
    pending_question = st.session_state.pop("pending_question", None)
    if pending_question:
        user_input = pending_question

    if user_input:
        # 添加用户消息
        if session_id not in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = []

        st.session_state.chat_history[session_id].append({
            "role": "user",
            "content": user_input,
        })

        # 显示用户消息
        with chat_container:
            display_message("user", user_input)

        # 生成回答
        with chat_container:
            with st.chat_message("assistant"):
                # 阶段1：检索中
                status_placeholder = st.empty()
                status_placeholder.info("🔍 正在检索相关文档...")
                message_placeholder = st.empty()
                full_response = ""
                sources = []
                final_response = None

                try:
                    import time
                    start_time = time.time()

                    # 流式生成
                    chunk_count = 0
                    for chunk, final_resp in pipeline.ask_stream(
                        question=user_input,
                        session_id=session_id,
                    ):
                        # 第一个 chunk 到达时，切换状态为生成中
                        if chunk_count == 0 and chunk:
                            status_placeholder.success("✅ 检索完成，正在生成回答...")
                            time.sleep(0.3)
                            status_placeholder.empty()

                        if final_resp is not None:
                            # 最终响应，包含引用来源
                            final_response = final_resp
                            sources = [s.__dict__ for s in final_resp.sources]
                        else:
                            # 流式内容
                            full_response += chunk
                            chunk_count += 1
                            if full_response:
                                message_placeholder.markdown(full_response + "▌")

                    # 如果没有收到任何流式 chunk（智谱可能整段返回），用最终响应
                    if not full_response and final_response:
                        full_response = final_response.answer

                    # 最终显示
                    message_placeholder.markdown(full_response)
                    status_placeholder.empty()

                    elapsed = time.time() - start_time

                    # 保存助手消息
                    st.session_state.chat_history[session_id].append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    })

                    # 显示耗时
                    st.caption(f"⏱️ 耗时: {elapsed:.1f}s | 检索到 {len(sources)} 个引用来源")

                    # 显示引用来源
                    if sources:
                        with st.expander(f"📚 引用来源 ({len(sources)} 个)"):
                            for source in sources:
                                source_info = source.get("filename", "未知文件")
                                if source.get("page"):
                                    source_info += f" · 第{source['page']}页"
                                if source.get("sheet"):
                                    source_info += f" · Sheet: {source['sheet']}"

                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-title">[来源{source.get('index', '?')}] {source_info}</div>
                                        <div class="source-meta">相关度: {source.get('score', 0):.2%}</div>
                                        <div class="source-snippet">"{source.get('snippet', '')}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                except Exception as e:
                    error_msg = f"❌ 生成回答时出错: {str(e)}"
                    status_placeholder.error(error_msg)
                    message_placeholder.empty()
                    logger.error(f"问答失败: {str(e)}")

                    # 保存错误消息
                    st.session_state.chat_history[session_id].append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    })

                    # 显示详细错误信息（帮助调试）
                    with st.expander("🔧 查看详细错误信息"):
                        import traceback
                        st.code(traceback.format_exc())

        st.rerun()


if __name__ == "__main__":
    main()
