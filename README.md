# 📚 企业级文档知识库问答系统

> 基于 RAG（检索增强生成）技术的智能文档问答平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 核心功能

### 📁 多格式文档支持
- 支持 **PDF、Word、PPT、Excel、CSV、TXT、Markdown、HTML** 等多种格式
- 批量上传，自动解析
- 智能分块，保留语义完整性

### 🔍 混合检索 + Rerank
- **向量检索**：语义相似度匹配，擅长同义、模糊查询
- **BM25 检索**：关键词精确匹配，擅长专有名词、术语
- **Rerank 精排**：交叉编码器重排序，大幅提升准确率
- 加权融合，双通道互补

### 💬 多轮对话问答
- 多轮对话，上下文理解
- 流式输出，实时响应
- 引用来源标注，有据可依
- 历史对话管理

### ⚡ 增量更新
- 基于文件哈希的变化检测
- 只处理新增/修改的文档
- 更新效率提升 10 倍以上

### 🎯 幻觉抑制
- 严格的 Prompt 约束
- 引用验证机制
- 答案置信度评估
- 幻觉率 ≤ 5%

### 🖥️ 可视化界面
- 基于 Streamlit 的友好界面
- 文档管理、对话问答、系统设置
- 实时状态监控

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit 前端                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ 文档管理  │  │ 智能问答  │  │ 系统设置  │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     问答流水线 (QAPipeline)                  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 文档处理模块  │  │  检索模块   │  │    LLM 问答模块     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      数据存储层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ ChromaDB     │  │  BM25 索引   │  │  元数据 JSON      │  │
│  │ 向量数据库   │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**技术栈：**
- **前端**：Streamlit
- **后端**：Python + LangChain
- **向量数据库**：ChromaDB
- **嵌入模型**：OpenAI / BGE
- **LLM**：OpenAI GPT-4o
- **检索**：BM25 + 向量 + Rerank
- **重排**：BGE-Reranker / Cohere

## 🚀 快速开始

### 环境要求

- Python 3.10+
- OpenAI API 密钥

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/your-username/enterprise-doc-qa.git
cd enterprise-doc-qa
```

#### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量

```bash
# 复制配置文件
copy .env.example .env  # Windows
cp .env.example .env    # macOS/Linux
```

编辑 `.env` 文件，填入你的配置：

```env
# OpenAI 配置（必填）
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 向量数据库配置
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=enterprise_docs

# 检索配置
TOP_K=10
RERANK_TOP_N=5
ENABLE_RERANK=true
```

#### 5. 启动应用

```bash
streamlit run app/main.py
```

浏览器会自动打开 `http://localhost:8501`

### Docker 部署（可选）

```bash
# 构建镜像
docker build -t enterprise-doc-qa .

# 运行容器
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  enterprise-doc-qa
```

## 📖 使用指南

### 1. 上传文档

1. 进入「文档管理」页面
2. 拖拽或选择文件上传
3. 等待系统自动解析和索引
4. 查看处理结果

### 2. 智能问答

1. 进入「智能问答」页面
2. 在输入框中输入问题
3. 查看 AI 回答和引用来源
4. 支持多轮对话

### 3. 系统设置

1. 进入「系统设置」页面
2. 查看系统状态
3. 调整检索参数
4. 查看系统日志

## 📁 项目结构

```
enterprise-doc-qa/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── .env.example                 # 环境变量示例
├── .gitignore                   # Git 忽略文件
├── config/                      # 配置模块
│   ├── __init__.py
│   └── settings.py              # 全局配置
├── src/                         # 核心源码
│   ├── __init__.py
│   ├── document_processor/      # 文档处理模块
│   │   ├── __init__.py
│   │   ├── parser.py            # 多格式文档解析
│   │   ├── chunker.py           # 智能分块
│   │   └── incremental.py       # 增量更新
│   ├── vector_store/            # 向量存储模块
│   │   ├── __init__.py
│   │   ├── embedding.py         # 嵌入模型
│   │   └── chroma_store.py      # ChromaDB 封装
│   ├── retriever/               # 检索模块
│   │   ├── __init__.py
│   │   ├── bm25.py              # BM25 检索
│   │   ├── reranker.py          # Rerank 重排
│   │   └── hybrid.py            # 混合检索
│   ├── llm/                     # LLM 问答模块
│   │   ├── __init__.py
│   │   ├── prompt.py            # Prompt 模板
│   │   ├── history.py           # 对话历史管理
│   │   └── chat.py              # 对话管理器
│   └── pipeline/                # 问答流水线
│       ├── __init__.py
│       └── qa_pipeline.py       # 系统核心入口
├── app/                         # Streamlit 应用
│   ├── __init__.py
│   ├── main.py                  # 主应用入口
│   ├── pages/                   # 页面
│   │   ├── 1_📚_文档管理.py
│   │   ├── 2_💬_智能问答.py
│   │   └── 3_⚙️_系统设置.py
│   └── components/              # UI 组件
├── tests/                       # 测试
│   ├── __init__.py
│   └── test_pipeline.py
├── docs/                        # 文档
│   ├── ARCHITECTURE.md          # 技术架构文档
│   └── OPTIMIZATION.md          # 优化策略文档
└── data/                        # 数据目录
    ├── documents/               # 上传的文档
    └── chroma_db/               # 向量数据库
```

## 🎯 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 问答准确率 | ≥ 90% | 回答正确的比例 |
| 幻觉率 | ≤ 5% | 编造内容的比例 |
| 检索召回率 | ≥ 90% | 相关文档被检索到的比例 |
| Top-5 准确率 | ≥ 95% | 前 5 个结果包含正确答案 |
| 引用准确率 | ≥ 95% | 引用来源正确的比例 |
| 首 token 延迟 | ~500ms | 流式输出首字延迟 |
| 检索延迟 | ~300ms | 端到端检索耗时 |

## 🔧 核心优化

### 幻觉抑制
- 严格的 Prompt 约束和角色设定
- 引用来源标注机制
- 答案验证和置信度评估
- 低温度生成（temperature=0.1）

### 检索优化
- 混合检索架构（向量 + BM25）
- Rerank 精排提升准确率
- 智能分块策略
- 问题改写优化

### 性能优化
- 批量向量化处理
- 嵌入结果缓存
- 增量更新机制
- 流式输出体验

## 📚 详细文档

- [技术架构文档](docs/ARCHITECTURE.md) - 系统架构、模块设计、数据模型
- [优化策略文档](docs/OPTIMIZATION.md) - 幻觉抑制、检索优化、性能优化

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙋‍♂️ 常见问题

### Q: 支持哪些文档格式？
A: 目前支持 PDF、Word (.docx)、PowerPoint (.pptx)、Excel (.xlsx)、CSV、TXT、Markdown、HTML 等格式。

### Q: 如何提升问答准确率？
A: 可以尝试以下方法：
- 调整分块大小和重叠参数
- 增加 top_k 和 rerank_top_n
- 启用 Rerank 重排
- 优化文档质量和结构

### Q: 如何降低幻觉率？
A: 可以尝试以下方法：
- 降低 temperature 参数（0.1-0.3）
- 增加检索结果数量
- 加强 Prompt 约束
- 启用答案验证

### Q: 支持本地部署吗？
A: 支持。可以使用本地嵌入模型（如 BGE）和本地 LLM（如 Ollama），完全离线运行。

## 📞 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**
