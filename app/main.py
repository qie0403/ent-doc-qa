"""
企业级文档知识库问答系统 - Streamlit 应用入口
Enterprise Document Knowledge Base Q&A System - Streamlit Entry

功能说明：
    统一入口文件（保持启动命令为 streamlit run app/main.py），
    使用 st.navigation 显式定义侧边栏导航与页面标题，
    各页面实现位于 app/pages/ 目录。

运行方式：
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# 全局页面配置（页面标题、图标、布局）
st.set_page_config(
    page_title="DocMind 智析 · 企业文档知识库问答平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 显式定义导航：文件名保持稳定，展示名与图标在这里统一配置
home_page = st.Page(
    "pages/home.py",
    title="首页",
    icon="🏠",
    default=True,
)
document_page = st.Page(
    "pages/1_📚_文档管理.py",
    title="文档管理",
    icon="📚",
)
qa_page = st.Page(
    "pages/2_💬_智能问答.py",
    title="智能问答",
    icon="💬",
)
settings_page = st.Page(
    "pages/3_⚙️_系统设置.py",
    title="系统设置",
    icon="⚙️",
)

nav = st.navigation(
    [home_page, document_page, qa_page, settings_page],
    position="sidebar",
)
nav.run()
