# 技术架构文档

## 一、系统架构

### 1.1 整体架构

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

### 1.2 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| 前端 | Streamlit | 快速构建数据应用界面 |
| 后端 | Python + LangChain | 业务逻辑和 LLM 编排 |
| 向量数据库 | ChromaDB | 轻量级本地向量数据库 |
| 嵌入模型 | OpenAI / BGE | 文本向量化 |
| LLM | OpenAI GPT-4o | 对话生成 |
| 检索 | BM25 + 向量 + Rerank | 混合检索架构 |
| 重排 | BGE-Reranker / Cohere | 检索结果精排 |
| 文档解析 | pypdf / python-docx / pandas | 多格式文档解析 |

## 二、核心模块详解

### 2.1 文档处理模块 (document_processor)

**职责：** 文档解析、智能分块、增量更新

**子模块：**
- `parser.py`：多格式文档解析器
  - 支持 PDF、Word、PPT、Excel、CSV、TXT、Markdown、HTML
  - 统一输出 LangChain Document 对象
  - 丰富的元数据（文件名、页码、段落号等）

- `chunker.py`：智能文档分块器
  - 递归字符分块（默认）
  - Token 分块
  - Markdown 标题分块
  - 自适应分块（按文档类型选择策略）

- `incremental.py`：增量更新管理器
  - 基于 MD5 文件哈希的变化检测
  - 新增/修改/删除文件识别
  - 元数据持久化存储

### 2.2 向量存储模块 (vector_store)

**职责：** 文本向量化、向量存储与检索

**子模块：**
- `embedding.py`：嵌入模型管理器
  - 支持 OpenAI API 嵌入
  - 支持本地 Sentence-Transformers
  - 批量嵌入优化
  - 结果缓存机制

- `chroma_store.py`：ChromaDB 向量数据库封装
  - 文档增删改查
  - 相似度搜索
  - 元数据过滤
  - 集合管理
  - 持久化存储

### 2.3 检索模块 (retriever)

**职责：** 混合检索、Rerank 精排

**子模块：**
- `bm25.py`：BM25 关键词检索器
  - 中文分词（jieba）
  - 停用词过滤
  - BM25Okapi 算法
  - 增量索引更新

- `reranker.py`：重排器
  - BGE Reranker（本地部署）
  - Cohere Rerank（API 调用）
  - 统一接口，可切换
  - 降级策略

- `hybrid.py`：混合检索器
  - 双通道检索（向量 + BM25）
  - 结果去重合并
  - 加权分数融合
  - Rerank 精排集成

### 2.4 LLM 问答模块 (llm)

**职责：** 对话管理、Prompt 构建、流式输出

**子模块：**
- `prompt.py`：Prompt 模板工厂
  - 基础问答 Prompt
  - 多轮对话 Prompt
  - 问题改写 Prompt
  - 幻觉检测 Prompt
  - 上下文格式化工具

- `history.py`：对话历史管理器
  - 多会话管理
  - 滑动窗口截断
  - Token 数量控制
  - 会话持久化

- `chat.py`：对话管理器
  - 单轮/多轮问答
  - 流式输出
  - 问题改写优化
  - 引用来源提取
  - 错误处理

### 2.5 问答流水线 (pipeline)

**职责：** 系统核心编排，统一入口

**核心类：** `QAPipeline`

**主要功能：**
- 系统初始化和组件管理
- 文档管理 API（上传、删除、批量导入）
- 问答 API（流式/非流式）
- 系统状态查询
- 健康检查

## 三、核心流程

### 3.1 文档入库流程

```
用户上传文档
    │
    ▼
文档解析 (DocumentParser)
    ├─ PDF → pypdf 提取文本
    ├─ Word → python-docx 提取段落
    ├─ Excel → pandas 按行处理
    └─ ...
    │
    ▼
智能分块 (AdaptiveChunker)
    ├─ 按文档类型选择分块策略
    ├─ 语义边界分割
    └─ 重叠分块
    │
    ▼
向量化 (EmbeddingManager)
    ├─ 批量嵌入
    └─ 缓存优化
    │
    ▼
存入向量数据库 (ChromaVectorStore)
    ├─ 文本 + 向量 + 元数据
    └─ 持久化存储
    │
    ▼
更新 BM25 索引
    │
    ▼
更新增量元数据
    │
    ▼
完成
```

### 3.2 问答流程

```
用户提问
    │
    ▼
问题改写 (可选)
    ├─ 结合对话历史
    └─ 优化问题表述
    │
    ▼
混合检索 (HybridRetriever)
    ├─ 向量检索 → top_k 结果
    ├─ BM25 检索 → top_k 结果
    ├─ 去重合并 + 加权融合
    └─ Rerank 精排 → top_n 结果
    │
    ▼
构建 Prompt
    ├─ 系统指令
    ├─ 参考文档上下文
    ├─ 对话历史 (多轮)
    └─ 用户问题
    │
    ▼
LLM 生成回答
    ├─ 流式输出
    └─ 引用标注
    │
    ▼
解析引用来源
    │
    ▼
保存对话历史
    │
    ▼
返回结果
```

## 四、数据模型

### 4.1 Document (LangChain)

```python
Document(
    page_content: str,          # 文档内容
    metadata: Dict[str, Any]    # 元数据
)
```

**元数据字段：**
- `source`：文件路径
- `filename`：文件名
- `file_type`：文件类型
- `page`：页码（PDF/PPT）
- `paragraph`：段落号（Word）
- `sheet`：工作表名（Excel）
- `row`：行号（Excel/CSV）
- `chunk_index`：分块索引
- `total_chunks`：总分块数

### 4.2 RetrievalResult

```python
RetrievalResult(
    document: Document,         # 文档
    score: float,               # 最终分数
    rank: int,                  # 排名
    source: str,                # 来源：vector/bm25/hybrid
    vector_score: float,        # 向量检索分数
    bm25_score: float,          # BM25 检索分数
    rerank_score: float         # 重排分数
)
```

### 4.3 ChatResponse

```python
ChatResponse(
    answer: str,                        # 回答内容
    sources: List[SourceCitation],      # 引用来源列表
    retrieved_docs: List[RetrievalResult],  # 检索到的文档
    session_id: str,                    # 会话 ID
    latency_ms: float,                  # 耗时（毫秒）
    metadata: Dict[str, Any]            # 其他元数据
)
```

## 五、配置说明

### 5.1 核心配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| OPENAI_API_KEY | - | OpenAI API 密钥 |
| OPENAI_MODEL | gpt-4o | 对话模型 |
| OPENAI_EMBEDDING_MODEL | text-embedding-3-small | 嵌入模型 |
| CHUNK_SIZE | 512 | 分块大小（字符） |
| CHUNK_OVERLAP | 64 | 分块重叠（字符） |
| TOP_K | 10 | 初始检索数量 |
| RERANK_TOP_N | 5 | 重排后数量 |
| ENABLE_RERANK | true | 是否启用重排 |
| MAX_HISTORY_TURNS | 10 | 最大历史轮数 |
| TEMPERATURE | 0.1 | 生成温度 |

### 5.2 配置文件

- `.env`：环境变量配置
- `config/settings.py`：配置类定义
- 支持运行时动态调整部分参数

## 六、扩展点

### 6.1 新增文档格式

1. 在 `DocumentParser` 中添加新的解析方法
2. 注册到 `supported_formats` 字典
3. 在 `AdaptiveChunker` 中添加对应的分块配置

### 6.2 新增向量数据库

1. 继承基类实现新的向量存储类
2. 实现 `add_documents`、`similarity_search` 等方法
3. 在 `QAPipeline` 中替换默认实现

### 6.3 新增重排模型

1. 继承 `Reranker` 基类
2. 实现 `rerank` 方法
3. 在 `create_reranker` 工厂函数中注册

### 6.4 新增 LLM 提供商

1. 使用 LangChain 对应的 Chat Model 类
2. 在 `QAPipeline._init_llm` 中添加初始化逻辑
3. 支持配置切换

## 七、性能指标

### 7.1 检索性能

- 向量检索：~50ms（10k 文档规模）
- BM25 检索：~20ms
- Rerank 精排：~200ms（BGE-base）
- 端到端检索：~300ms

### 7.2 问答性能

- 首 token 延迟：~500ms
- 生成速度：~30 tokens/s
- 端到端延迟：~3-5s（取决于回答长度）

### 7.3 准确率指标

- 检索召回率：≥ 90%
- Top-5 准确率：≥ 95%
- 回答准确率：≥ 90%
- 幻觉率：≤ 5%
