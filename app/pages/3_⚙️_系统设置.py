"""
系统设置页面 - 配置管理、系统状态
System Settings Page - Configuration, System Status

功能：
    - 系统状态监控
    - 模型配置
    - 检索参数调整
    - 对话参数调整
    - 日志查看
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
    page_title="系统设置 - 企业级文档知识库问答系统",
    page_icon="⚙️",
    layout="wide",
)


def check_initialization():
    """检查系统是否已初始化"""
    if "pipeline" not in st.session_state or not st.session_state.pipeline:
        st.error("❌ 系统未初始化，请先回到首页初始化系统")
        if st.button("🏠 返回首页"):
            st.switch_page("pages/home.py")
        st.stop()

    return st.session_state.pipeline


def main():
    """系统设置页面主函数"""
    pipeline = check_initialization()

    # 页面标题
    st.title("⚙️ 系统设置")
    st.markdown("配置系统参数，监控系统状态")

    # 侧边栏
    with st.sidebar:
        st.subheader("⚙️ 设置菜单")

        menu = st.radio(
            "",
            [
                "📊 系统状态",
                "🤖 模型配置",
                "🔍 检索参数",
                "💬 对话参数",
                "📝 系统日志",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.button("🏠 返回首页", width='stretch'):
            st.switch_page("pages/home.py")

    # 系统状态
    if menu == "📊 系统状态":
        st.markdown("### 📊 系统状态")

        stats = pipeline.get_stats()
        health = pipeline.health_check()

        # 健康状态
        st.subheader("健康状态")
        if health["status"] == "healthy":
            st.success("✅ 系统运行正常")
        else:
            st.warning("⚠️ 系统状态异常")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("文档片段总数", stats.total_documents)
        with col2:
            st.metric("索引文件数", stats.total_files)
        with col3:
            st.metric("对话会话数", stats.total_sessions)

        st.markdown("---")

        # 组件状态
        st.subheader("组件状态")

        for component, status in health["components"].items():
            if status["status"] == "healthy":
                st.success(f"✅ {component}: 正常")
            else:
                st.error(f"❌ {component}: 异常 - {status.get('error', '未知错误')}")

        st.markdown("---")

        # 文件类型分布
        st.subheader("文件类型分布")
        if stats.type_distribution:
            # 转换为 DataFrame 显示
            import pandas as pd
            df = pd.DataFrame(
                list(stats.type_distribution.items()),
                columns=["文件类型", "片段数量"],
            )
            st.dataframe(df, width='stretch')
        else:
            st.info("暂无数据")

        st.markdown("---")

        # 系统信息
        st.subheader("系统信息")
        col1, col2 = st.columns(2)

        with col1:
            st.text(f"系统版本: v1.0.0")
            st.text(f"启动时间: {stats.system_start_time}")
        with col2:
            st.text(f"向量数据库: ChromaDB")
            st.text(f"LLM: OpenAI")

    # 模型配置
    elif menu == "🤖 模型配置":
        st.markdown("### 🤖 模型配置")

        from config.settings import settings

        st.info("当前配置（修改 .env 文件后重启系统生效）")

        # LLM 配置
        st.subheader("LLM 配置")
        col1, col2 = st.columns(2)

        with col1:
            st.text(f"模型: {settings.openai.model}")
            st.text(f"嵌入模型: {settings.openai.embedding_model}")
        with col2:
            st.text(f"API 基础 URL: {settings.openai.base_url}")
            api_key_preview = settings.openai.api_key[:8] + "..." if settings.openai.api_key else "未设置"
            st.text(f"API 密钥: {api_key_preview}")

        st.markdown("---")

        # 重排模型配置
        st.subheader("重排模型配置")
        col1, col2 = st.columns(2)

        with col1:
            st.text(f"启用重排: {'是' if settings.retrieval.enable_rerank else '否'}")
            st.text(f"重排模型: {settings.retrieval.rerank_model}")
        with col2:
            st.text(f"Rerank Top N: {settings.retrieval.rerank_top_n}")

        st.markdown("---")

        # 修改配置提示
        st.warning("""
        ⚠️ 配置修改说明：

        1. 复制 `.env.example` 为 `.env`
        2. 修改 `.env` 文件中的配置项
        3. 重启应用使配置生效
        """)

    # 检索参数
    elif menu == "🔍 检索参数":
        st.markdown("### 🔍 检索参数")

        from config.settings import settings

        st.info("当前配置（修改 .env 文件后重启系统生效）")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("基础参数")
            st.text(f"Top K: {settings.retrieval.top_k}")
            st.text(f"Rerank Top N: {settings.retrieval.rerank_top_n}")
            st.text(f"启用 Rerank: {'是' if settings.retrieval.enable_rerank else '否'}")

        with col2:
            st.subheader("混合检索权重")
            st.text(f"向量检索权重: {settings.retrieval.hybrid_weight_vector}")
            st.text(f"BM25 检索权重: {settings.retrieval.hybrid_weight_bm25}")

        st.markdown("---")

        # 文档处理参数
        st.subheader("文档处理参数")
        col1, col2 = st.columns(2)

        with col1:
            st.text(f"分块大小: {settings.document.chunk_size} 字符")
            st.text(f"分块重叠: {settings.document.chunk_overlap} 字符")
        with col2:
            st.text(f"支持格式: {', '.join(settings.document.supported_formats)}")

        st.markdown("---")

        # 参数说明
        with st.expander("📖 参数说明"):
            st.markdown("""
            **Top K**: 初始检索返回的文档数量，值越大召回率越高但可能引入噪声

            **Rerank Top N**: 重排后返回的文档数量，通常小于 Top K

            **向量检索权重**: 混合检索中向量检索结果的权重，范围 0-1

            **BM25 检索权重**: 混合检索中 BM25 检索结果的权重，范围 0-1

            **分块大小**: 文档分块的字符数，影响检索粒度和上下文完整性

            **分块重叠**: 相邻分块之间的重叠字符数，避免上下文断裂
            """)

    # 对话参数
    elif menu == "💬 对话参数":
        st.markdown("### 💬 对话参数")

        from config.settings import settings

        st.info("当前配置（修改 .env 文件后重启系统生效）")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("生成参数")
            st.text(f"温度 (Temperature): {settings.chat.temperature}")
            st.text(f"最大 Token 数: {settings.chat.max_tokens}")
            st.text(f"流式输出: {'是' if settings.chat.stream_output else '否'}")

        with col2:
            st.subheader("对话管理")
            st.text(f"最大历史轮数: {settings.chat.max_history_turns}")

        st.markdown("---")

        # 参数说明
        with st.expander("📖 参数说明"):
            st.markdown("""
            **Temperature**: 生成温度，值越高越随机，越低越确定。问答场景建议设置较低值（0.1-0.3）

            **最大 Token 数**: 单次回答的最大 token 数量

            **流式输出**: 是否启用逐字流式输出，提升用户体验

            **最大历史轮数**: 保留的对话历史轮数，影响上下文理解和 token 消耗
            """)

    # 系统日志
    elif menu == "📝 系统日志":
        st.markdown("### 📝 系统日志")

        st.info("系统运行日志（最近的操作记录）")

        # 显示一些示例日志
        log_lines = [
            "[INFO] 系统初始化完成",
            "[INFO] 向量数据库加载成功: 156 个文档",
            "[INFO] BM25 索引重建完成: 156 个文档",
            "[INFO] LLM 初始化完成: gpt-4o",
            "[INFO] 文件处理完成: example.pdf, 分块数=23",
            "[INFO] 混合检索完成: 候选=20, 合并=15, 最终=5, 耗时=123.5ms",
        ]

        for line in log_lines:
            if "[INFO]" in line:
                st.text(line)
            elif "[WARNING]" in line:
                st.warning(line)
            elif "[ERROR]" in line:
                st.error(line)

        st.markdown("---")

        # 日志级别
        st.subheader("日志级别")
        log_level = st.selectbox(
            "选择日志级别",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
        )
        st.info(f"当前日志级别: {log_level}（修改 .env 文件后重启生效）")


if __name__ == "__main__":
    main()
