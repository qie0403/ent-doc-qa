"""
按当前 .env / 环境变量中的 CHUNK_SIZE 全量重建知识库
流程: 确保语料就位 -> 清空向量库 -> 重新解析 data/documents -> 分块 -> 向量化入库

语料来源：
    data/documents 为空时，自动从入库的 sample_data/ 复制示例语料，
    保证新克隆的仓库可直接重建出可用的演示知识库。

运行:
    $env:CHUNK_SIZE=256; python scripts/reindex.py
    $env:CHUNK_SIZE=512; python scripts/reindex.py
"""
import shutil
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.pipeline import QAPipeline

SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def ensure_corpus(docs_dir: Path) -> Path:
    """确保语料目录非空：为空时从 sample_data/ 复制示例语料"""
    docs_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        p for p in docs_dir.iterdir()
        if p.is_file() and p.name != ".gitkeep"
    ]
    if existing:
        print(f"语料目录已有 {len(existing)} 个文件，直接使用: {docs_dir}")
        return docs_dir

    if not SAMPLE_DATA_DIR.is_dir():
        print(f"[警告] 语料目录为空且未找到示例语料目录: {SAMPLE_DATA_DIR}")
        return docs_dir

    copied = 0
    for path in sorted(SAMPLE_DATA_DIR.glob("*.txt")):
        shutil.copy2(path, docs_dir / path.name)
        copied += 1
        print(f"  已从 sample_data 复制: {path.name}")
    print(f"共复制 {copied} 个示例语料文件到 {docs_dir}")
    return docs_dir


def main():
    print(f"当前分块配置: CHUNK_SIZE={settings.document.chunk_size}, "
          f"CHUNK_OVERLAP={settings.document.chunk_overlap}")

    docs_dir = PROJECT_ROOT / "data" / "documents"
    ensure_corpus(docs_dir)

    pipeline = QAPipeline()

    print("清空旧索引...")
    pipeline.clear_knowledge_base()

    print(f"全量重建索引: {docs_dir}")
    result = pipeline.add_directory(docs_dir, incremental=False)
    print(f"处理结果: 成功文件={result['success_count']}, "
          f"总分块数={result['total_chunks']}, 失败={result['failed_count']}")
    if result["failed_files"]:
        print("失败明细:", result["failed_files"])

    stats = pipeline.get_stats()
    print(f"向量库最终文档片段数: {stats.total_documents}")


if __name__ == "__main__":
    main()
