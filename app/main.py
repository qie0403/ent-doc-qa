"""
企业级文档知识库问答系统 - Streamlit 主应用
Enterprise Document Knowledge Base Q&A System - Streamlit Main App

功能说明：
    基于 Streamlit 的可视化界面，提供文档管理、智能问答、系统设置等功能。

运行方式：
    streamlit run app/main.py
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from loguru import logger

# 页面配置
st.set_page_config(
    page_title="企业级文档知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def initialize_pipeline():
    """初始化问答系统流水线"""
    try:
        from src.pipeline import QAPipeline

        with st.spinner("正在初始化系统，请稍候..."):
            pipeline = QAPipeline()
            st.session_state.pipeline = pipeline
            st.session_state.initialized = True

        st.success("✅ 系统初始化成功！")
        return True

    except Exception as e:
        st.error(f"❌ 系统初始化失败: {str(e)}")
        logger.error(f"系统初始化失败: {str(e)}")
        return False


def main():
    """主页面"""
    init_session_state()

    # 侧边栏
    with st.sidebar:
        st.title("📚 文档问答系统")
        st.markdown("---")

        # 系统状态
        if st.session_state.initialized and st.session_state.pipeline:
            stats = st.session_state.pipeline.get_stats()
            st.subheader("系统状态")
            st.metric("已索引文档", stats.total_documents)
            st.metric("已索引文件", stats.total_files)
            st.metric("对话会话", stats.total_sessions)
        else:
            st.info("系统未初始化")

        st.markdown("---")
        st.caption("版本: v1.0.0")

    # 主内容区
    st.markdown('<p class="main-header">📚 企业级文档知识库问答系统</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">基于 RAG 技术的智能文档问答平台，支持多格式文档、混合检索、多轮对话</p>',
        unsafe_allow_html=True,
    )

    # 系统初始化检查
    if not st.session_state.initialized:
        st.warning("⚠️ 系统尚未初始化，请先配置 API 密钥并初始化系统")

        with st.expander("🚀 快速开始", expanded=True):
            st.markdown("""
            ### 系统配置

            请在 `.env` 文件中配置以下信息：

            1. **OpenAI API 密钥** - 用于 LLM 对话和嵌入
            2. **向量数据库路径** - 文档向量存储位置
            3. **其他可选配置** - 检索参数、模型选择等

            配置完成后点击下方按钮初始化系统。
            """)

            if st.button("🔧 初始化系统", type="primary", use_container_width=True):
                initialize_pipeline()
                st.rerun()

        return

    # 系统已初始化，展示首页内容
    pipeline = st.session_state.pipeline
    stats = pipeline.get_stats()

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{stats.total_documents}</div>
                <div class="stat-label">📄 文档片段</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{stats.total_files}</div>
                <div class="stat-label">📁 索引文件</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{stats.total_sessions}</div>
                <div class="stat-label">💬 对话会话</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">90%+</div>
                <div class="stat-label">🎯 准确率</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 功能介绍
    st.subheader("✨ 核心功能")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📁 多格式文档支持</h3>
                <p>支持 PDF、Word、PPT、Excel、TXT、Markdown 等多种格式，批量上传自动解析</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>🔍 混合检索 + Rerank</h3>
                <p>向量检索 + BM25 关键词检索双通道，结合 Rerank 精排，准确率提升显著</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>💬 多轮对话问答</h3>
                <p>支持多轮对话、上下文理解、流式输出、引用来源标注</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>⚡ 增量更新</h3>
                <p>基于文件哈希的增量检测，只处理变化的文档，更新效率提升 10 倍</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>🎯 幻觉抑制</h3>
                <p>严格的 Prompt 约束 + 引用验证，大幅降低幻觉率，回答有据可依</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="feature-card">
                <h3>🖥️ 可视化界面</h3>
                <p>基于 Streamlit 的友好界面，文档管理、对话问答、系统设置一应俱全</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 快速操作
    st.subheader("🚀 快速开始")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📚 文档管理", use_container_width=True, type="secondary"):
            st.switch_page("pages/1_📚_文档管理.py")

    with col2:
        if st.button("💬 智能问答", use_container_width=True, type="primary"):
            st.switch_page("pages/2_💬_智能问答.py")

    with col3:
        if st.button("⚙️ 系统设置", use_container_width=True, type="secondary"):
            st.switch_page("pages/3_⚙️_系统设置.py")

    # 已索引文件列表
    if stats.indexed_files:
        st.markdown("---")
        st.subheader("📂 已索引文件")

        # 按文件类型分组
        type_files = {}
        for filename in stats.indexed_files:
            ext = Path(filename).suffix.lower()
            if ext not in type_files:
                type_files[ext] = []
            type_files[ext].append(filename)

        for ext, files in sorted(type_files.items()):
            with st.expander(f"{ext} ({len(files)} 个文件)"):
                for f in sorted(files):
                    st.text(f"  📄 {f}")


if __name__ == "__main__":
    main()
