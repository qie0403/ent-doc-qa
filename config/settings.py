"""
企业级文档知识库问答系统 - 全局配置模块
Enterprise Document Knowledge Base Q&A System - Global Configuration

功能说明：
    基于 Pydantic Settings 的类型安全配置管理，支持从 .env 文件加载配置。
    所有模块统一从此处读取配置，确保配置来源单一、可维护。

技术选型：
    - pydantic-settings：类型安全的配置管理，支持环境变量、.env 文件
    - 分层配置：按功能模块组织配置项，便于扩展和维护
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 优先加载 .env 文件到环境变量（确保嵌套配置能正确读取）
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)


class OpenAIConfig(BaseSettings):
    """OpenAI API 配置（兼容智谱 AI 等 OpenAI 格式接口）"""
    api_key: str = Field(default="", description="API 密钥")
    base_url: str = Field(default="https://api.openai.com/v1", description="API 基础 URL")
    model: str = Field(default="gpt-4o", description="对话模型名称")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="嵌入模型名称"
    )

    model_config = SettingsConfigDict(env_prefix="OPENAI_", extra="ignore")


class ChromaConfig(BaseSettings):
    """ChromaDB 向量数据库配置"""
    persist_dir: str = Field(default="./data/chroma_db", description="持久化存储目录")
    collection_name: str = Field(
        default="enterprise_docs",
        description="集合名称"
    )

    model_config = SettingsConfigDict(env_prefix="CHROMA_", extra="ignore")


class DocumentConfig(BaseSettings):
    """文档处理配置"""
    chunk_size: int = Field(default=512, description="分块大小（字符数）")
    chunk_overlap: int = Field(default=64, description="分块重叠大小")
    documents_dir: str = Field(default="./data/documents", description="文档存储目录")
    supported_formats: list = Field(
        default_factory=lambda: [".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".xlsx", ".csv"],
        description="支持的文档格式"
    )

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class RetrievalConfig(BaseSettings):
    """检索配置"""
    top_k: int = Field(default=10, description="初始检索返回数量")
    rerank_top_n: int = Field(default=5, description="重排后返回数量")
    # 默认关闭：开启需额外安装 FlagEmbedding 并下载本地模型，
    # 与 README / .env.example 的描述保持一致（避免新用户启动即报错）
    enable_rerank: bool = Field(default=False, description="是否启用重排")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-large",
        description="重排模型名称"
    )
    hybrid_weight_vector: float = Field(
        default=0.6,
        description="混合检索中向量检索权重"
    )
    hybrid_weight_bm25: float = Field(
        default=0.4,
        description="混合检索中 BM25 检索权重"
    )

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class ChatConfig(BaseSettings):
    """对话配置"""
    max_history_turns: int = Field(default=10, description="最大历史对话轮数")
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=2048, description="最大生成 token 数")
    stream_output: bool = Field(default=True, description="是否启用流式输出")

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class CohereConfig(BaseSettings):
    """Cohere Rerank 配置（可选）"""
    api_key: str = Field(default="", description="Cohere API 密钥")

    model_config = SettingsConfigDict(env_prefix="COHERE_", extra="ignore")


class LogConfig(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")


class Settings(BaseSettings):
    """全局配置类 - 所有配置的统一入口"""

    # 子配置
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    cohere: CohereConfig = Field(default_factory=CohereConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    # 项目基础配置
    project_name: str = "企业级文档知识库问答系统"
    version: str = "1.0.0"
    debug: bool = Field(default=False, description="调试模式")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def base_dir(self) -> Path:
        """获取项目根目录"""
        return Path(__file__).parent.parent.resolve()

    @property
    def chroma_persist_path(self) -> Path:
        """获取 ChromaDB 持久化路径（绝对路径）"""
        path = Path(self.chroma.persist_dir)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    @property
    def documents_path(self) -> Path:
        """获取文档目录（绝对路径）"""
        path = Path(self.document.documents_dir)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    def ensure_directories(self) -> None:
        """确保必要的目录存在"""
        self.chroma_persist_path.mkdir(parents=True, exist_ok=True)
        self.documents_path.mkdir(parents=True, exist_ok=True)


# 全局单例配置实例
settings = Settings()

# 确保目录存在
settings.ensure_directories()
