"""
文档管理页面 - 文档上传、管理、索引
Document Management Page - Upload, Manage, Index

功能：
    - 上传文档（支持多文件批量上传）
    - 查看已索引文件列表
    - 删除文档
    - 增量更新
    - 清空知识库
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import tempfile
import os
from loguru import logger

# 页面配置
st.set_page_config(
    page_title="文档管理 - 企业级文档知识库问答系统",
    page_icon="📚",
    layout="wide",
)

# 自定义 CSS
st.markdown("""
<style>
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }
    .file-item {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .file-name {
        font-weight: 500;
    }
    .file-meta {
        color: #666;
        font-size: 0.85rem;
    }
    .stat-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .badge-success {
        background: #d4edda;
        color: #155724;
    }
    .badge-warning {
        background: #fff3cd;
        color: #856404;
    }
    .badge-danger {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


def check_initialization():
    """检查系统是否已初始化"""
    if "pipeline" not in st.session_state or not st.session_state.pipeline:
        st.error("❌ 系统未初始化，请先回到首页初始化系统")
        if st.button("🏠 返回首页"):
            st.switch_page("main.py")
        st.stop()

    return st.session_state.pipeline


def save_uploaded_file(uploaded_file, save_dir: Path) -> Path:
    """
    保存上传的文件到临时目录

    Args:
        uploaded_file: Streamlit 上传文件对象
        save_dir: 保存目录

    Returns:
        保存后的文件路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def main():
    """文档管理页面主函数"""
    pipeline = check_initialization()

    # 页面标题
    st.title("📚 文档管理")
    st.markdown("上传、管理和索引您的文档，构建企业知识库")

    # 侧边栏
    with st.sidebar:
        st.subheader("📊 知识库统计")
        stats = pipeline.get_stats()

        st.metric("文档片段", stats.total_documents)
        st.metric("索引文件", stats.total_files)
        st.metric("对话会话", stats.total_sessions)

        st.markdown("---")

        # 文件类型分布
        if stats.type_distribution:
            st.subheader("📁 文件类型分布")
            for file_type, count in sorted(stats.type_distribution.items()):
                st.text(f"  {file_type}: {count} 个片段")

        st.markdown("---")

        if st.button("🏠 返回首页", use_container_width=True):
            st.switch_page("main.py")

    # 主内容区
    tab1, tab2, tab3 = st.tabs(["📤 上传文档", "📂 已索引文档", "⚡ 批量操作"])

    # Tab 1: 上传文档
    with tab1:
        st.markdown("### 上传文档")
        st.markdown("支持 PDF、Word、PPT、Excel、TXT、Markdown 等多种格式")

        # 上传区域
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "拖拽文件到此处，或点击选择文件",
            type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "csv", "txt", "md", "markdown", "html", "htm"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")

            # 显示文件列表
            for f in uploaded_files:
                file_size_mb = f.size / (1024 * 1024)
                st.markdown(
                    f"""
                    <div class="file-item">
                        <div>
                            <div class="file-name">📄 {f.name}</div>
                            <div class="file-meta">{file_size_mb:.2f} MB</div>
                        </div>
                        <span class="stat-badge badge-warning">待上传</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 上传按钮
            if st.button("🚀 开始上传并索引", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                success_count = 0
                failed_count = 0
                results = []

                # 创建临时目录
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"正在处理: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
                        progress_bar.progress((idx) / len(uploaded_files))

                        try:
                            # 保存文件
                            file_path = save_uploaded_file(uploaded_file, temp_path)

                            # 添加到知识库
                            result = pipeline.add_file(file_path)

                            if result["status"] == "success":
                                success_count += 1
                                results.append({
                                    "name": uploaded_file.name,
                                    "status": "success",
                                    "chunks": result["chunk_count"],
                                })
                            else:
                                failed_count += 1
                                results.append({
                                    "name": uploaded_file.name,
                                    "status": result["status"],
                                    "reason": result.get("reason", "unknown"),
                                })

                        except Exception as e:
                            failed_count += 1
                            results.append({
                                "name": uploaded_file.name,
                                "status": "failed",
                                "reason": str(e),
                            })
                            logger.error(f"处理文件失败: {uploaded_file.name}, {str(e)}")

                    progress_bar.progress(1.0)
                    status_text.text("处理完成！")

                # 显示结果
                st.markdown("---")
                st.subheader("📊 处理结果")

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ 成功: {success_count} 个文件")
                with col2:
                    if failed_count > 0:
                        st.error(f"❌ 失败: {failed_count} 个文件")
                    else:
                        st.info("全部处理成功！")

                # 详细结果
                with st.expander("查看详细结果"):
                    for r in results:
                        if r["status"] == "success":
                            st.markdown(
                                f"""
                                <div class="file-item">
                                    <div class="file-name">✅ {r['name']}</div>
                                    <span class="stat-badge badge-success">{r['chunks']} 个片段</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="file-item">
                                    <div>
                                        <div class="file-name">❌ {r['name']}</div>
                                        <div class="file-meta">{r.get('reason', '未知错误')}</div>
                                    </div>
                                    <span class="stat-badge badge-danger">失败</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # 刷新页面
                st.rerun()

    # Tab 2: 已索引文档
    with tab2:
        st.markdown("### 已索引文档")
        stats = pipeline.get_stats()

        if not stats.indexed_files:
            st.info("📭 知识库为空，请先上传文档")
        else:
            # 搜索框
            search_term = st.text_input("🔍 搜索文件", "")

            # 文件列表
            filtered_files = stats.indexed_files
            if search_term:
                filtered_files = [f for f in filtered_files if search_term.lower() in f.lower()]

            st.markdown(f"共 {len(filtered_files)} 个文件")

            for filename in sorted(filtered_files):
                col1, col2, col3 = st.columns([4, 1, 1])

                with col1:
                    st.markdown(f"📄 **{filename}**")

                with col2:
                    # 获取文件的分块数（简化处理）
                    st.text("已索引")

                with col3:
                    if st.button("🗑️ 删除", key=f"delete_{filename}", type="secondary"):
                        result = pipeline.delete_file(filename)
                        if result["status"] == "success":
                            st.success(f"已删除: {filename}")
                            st.rerun()
                        else:
                            st.error(f"删除失败: {result.get('error', '未知错误')}")

                st.markdown("---")

    # Tab 3: 批量操作
    with tab3:
        st.markdown("### 批量操作")

        # 从目录批量导入
        st.markdown("#### 📂 从目录批量导入")
        dir_path = st.text_input("输入目录路径", "")
        incremental = st.checkbox("增量更新（只处理变化的文件）", value=True)

        if st.button("🚀 开始批量导入", type="primary"):
            if not dir_path:
                st.warning("请输入目录路径")
            elif not Path(dir_path).exists():
                st.error("目录不存在")
            else:
                with st.spinner("正在批量处理..."):
                    result = pipeline.add_directory(dir_path, incremental=incremental)

                st.success("批量处理完成！")
                st.markdown(f"""
                - 发现文件: {result['total_files_found']} 个
                - 成功处理: {result['success_count']} 个
                - 处理失败: {result['failed_count']} 个
                - 跳过未变: {result['skipped_count']} 个
                - 删除文件: {result['deleted_count']} 个
                - 总分块数: {result['total_chunks']} 个
                """)

                if result['failed_files']:
                    with st.expander("查看失败文件"):
                        for fname, reason in result['failed_files']:
                            st.text(f"  ❌ {fname}: {reason}")

                st.rerun()

        st.markdown("---")

        # 清空知识库
        st.markdown("#### ⚠️ 清空知识库")
        st.warning("此操作将删除所有已索引的文档，不可恢复！")

        confirm = st.text_input("输入 'DELETE' 确认清空", "")
        if st.button("🗑️ 清空知识库", type="secondary", disabled=confirm != "DELETE"):
            with st.spinner("正在清空知识库..."):
                result = pipeline.clear_knowledge_base()

            if result["status"] == "success":
                st.success("知识库已清空")
                st.rerun()
            else:
                st.error("清空失败")


if __name__ == "__main__":
    main()
