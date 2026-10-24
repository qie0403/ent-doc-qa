"""
分块大小 A/B 对比实验：chunk_size=256 vs 512
=============================================

实验目的：
    验证 RAG 分块大小从 256 调整为 512 后，检索效果的变化，
    为"分块参数调优"提供可量化的实测数据（面试可直接引用）。

实验方法：
    1. 读取评测语料：优先使用入库的 sample_data/ 示例语料（新克隆仓库即可复现），
       若该目录缺失则回退为从当前 ChromaDB 中导出已索引语料并按文件还原全文（去重叠）；
    2. 分别用项目自带的 DocumentChunker（递归字符分块，分隔符与生产一致）
       按 256/32 与 512/64 两种参数重新分块（重叠均为 12.5%，与默认配置一致）；
    3. 分别构建项目自带的 BM25Retriever 索引（jieba 中文分词，与生产一致）；
    4. 用固定的 10 个问题测试集检索，统计：
       - 分块数 / 平均块长 / 最大块长
       - Recall@1、Recall@3（命中 = top-k 中出现正确文件且块内含答案关键词）
       - MRR（平均倒数排名）
       - 平均 Top1 BM25 分数

运行方式：
    python scripts/ab_chunk_size.py
"""

import sys
from pathlib import Path

# 项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.document_processor.chunker import DocumentChunker
from src.retriever.bm25 import BM25Retriever


# 入库的示例评测语料目录（保证新克隆仓库无需已有向量库即可复现本实验）
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


# ---------------------------------------------------------------------------
# 1. 读取评测语料
# ---------------------------------------------------------------------------

def load_corpus_from_sample_data():
    """从入库的 sample_data/ 读取语料全文（推荐来源：可复现、不依赖运行时数据）"""
    corpus = {}
    for path in sorted(SAMPLE_DATA_DIR.glob("*.txt")):
        corpus[path.name] = path.read_text(encoding="utf-8")
    return corpus


def load_corpus_from_chroma():
    """从 ChromaDB 读取全部 chunk，按文件 + chunk_index 拼接（去除相邻块重叠）"""
    import chromadb

    client = chromadb.PersistentClient(
        path=str(PROJECT_ROOT / "data" / "chroma_db")
    )
    collection = client.get_collection("enterprise_docs")
    data = collection.get()

    # 按文件分组
    files = {}
    for text, meta in zip(data["documents"], data["metadatas"]):
        filename = meta.get("filename", "unknown.txt")
        files.setdefault(filename, []).append(
            (meta.get("chunk_index", 0), text)
        )

    corpus = {}
    for filename, chunks in files.items():
        chunks.sort(key=lambda x: x[0])
        texts = [t for _, t in chunks]
        full_text = texts[0]
        for nxt in texts[1:]:
            # 去掉相邻块之间的重叠：在前一块尾部找与下一块头部相同的最长片段
            overlap = 0
            max_check = min(len(full_text), len(nxt), 200)
            for size in range(max_check, 0, -1):
                if full_text[-size:] == nxt[:size]:
                    overlap = size
                    break
            full_text += nxt[overlap:]
        corpus[filename] = full_text

    return corpus


def load_corpus():
    """加载评测语料：优先 sample_data/（可复现），缺失时回退 ChromaDB"""
    if SAMPLE_DATA_DIR.is_dir():
        corpus = load_corpus_from_sample_data()
        if corpus:
            print(f"[语料来源] sample_data/ 入库示例语料（{len(corpus)} 个文件）")
            return corpus
    print("[语料来源] ChromaDB 已索引语料（还原全文）")
    return load_corpus_from_chroma()


# ---------------------------------------------------------------------------
# 2. 测试问题集（问题 -> 正确文件 + 答案所在块必须包含的关键词）
#    关键词全部取自语料原文，保证评判客观
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    ("LangChain 框架的核心模块主要包括哪六个？", "LangChain.txt", "六个模块"),
    ("LangChain 中记忆模块的作用是什么？", "LangChain.txt", "记忆模块"),
    ("LangChain 的代理模块是如何完成任务的？", "LangChain.txt", "代理模块"),
    ("LangChain 的回调模块可以用来做什么？", "LangChain.txt", "回调模块"),
    ("LangChain 数据连接模块提供了哪些工具？", "LangChain.txt", "数据连接"),
    ("为什么说编写提示词时更长的提示不一定不好？", "prompt.txt", "更长"),
    ("想让模型输出便于程序解析，应该要求什么格式？", "prompt.txt", "结构化输出"),
    ("什么是“少量训练提示”？", "prompt.txt", "少量训练"),
    ("大模型的“幻觉”指的是什么？", "prompt.txt", "幻觉"),
    ("减少幻觉的一种有效策略是什么？", "prompt.txt", "引用"),
]


# ---------------------------------------------------------------------------
# 3. 单组实验
# ---------------------------------------------------------------------------

def run_experiment(corpus, chunk_size, chunk_overlap):
    """对给定分块参数执行一次完整实验，返回统计指标"""
    # 3.1 分块
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy="recursive",
    )
    docs = [
        Document(page_content=text, metadata={"filename": name})
        for name, text in corpus.items()
    ]
    chunks = chunker.split_documents(docs)

    chunk_lens = [len(c.page_content) for c in chunks]

    # 3.2 构建 BM25 索引
    retriever = BM25Retriever(language="chinese")
    retriever.add_documents(chunks)

    # 3.3 检索评测
    hits_at1 = 0
    hits_at3 = 0
    reciprocal_ranks = []
    top1_scores = []

    per_question = []
    for question, expected_file, keyword in TEST_QUESTIONS:
        results = retriever.search(question, top_k=10)
        top1_scores.append(results[0].score if results else 0.0)

        hit_rank = None
        for r in results:
            fname = r.document.metadata.get("filename", "")
            if fname == expected_file and keyword in r.document.page_content:
                hit_rank = r.rank
                break

        if hit_rank is not None:
            if hit_rank <= 1:
                hits_at1 += 1
            if hit_rank <= 3:
                hits_at3 += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        per_question.append({
            "question": question,
            "expected": expected_file,
            "hit_rank": hit_rank if hit_rank is not None else "未命中",
        })

    n = len(TEST_QUESTIONS)
    metrics = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "avg_chunk_len": sum(chunk_lens) / len(chunk_lens),
        "max_chunk_len": max(chunk_lens),
        "recall_at_1": hits_at1 / n,
        "recall_at_3": hits_at3 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "avg_top1_score": sum(top1_scores) / n,
        "per_question": per_question,
        "chunks": chunks,
    }
    return metrics


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------

def main():
    corpus = load_corpus()
    print("=" * 70)
    print("分块大小 A/B 实验：256 vs 512（重叠比例均为 12.5%）")
    print("=" * 70)
    for name, text in corpus.items():
        print(f"语料文件: {name} | 还原后全文字符数: {len(text)}")
    print(f"测试问题数: {len(TEST_QUESTIONS)}")
    print()

    result_256 = run_experiment(corpus, chunk_size=256, chunk_overlap=32)
    result_512 = run_experiment(corpus, chunk_size=512, chunk_overlap=64)

    def fmt_row(r):
        return (
            f"chunk_size={r['chunk_size']:>3} | 分块数={r['chunk_count']:>2} | "
            f"平均块长={r['avg_chunk_len']:>6.1f} | 最大块长={r['max_chunk_len']:>3} | "
            f"Recall@1={r['recall_at_1']:>5.1%} | Recall@3={r['recall_at_3']:>5.1%} | "
            f"MRR={r['mrr']:>5.3f} | 平均Top1分={r['avg_top1_score']:>6.2f}"
        )

    print("-" * 70)
    print("【总体指标对比】")
    print(fmt_row(result_256))
    print(fmt_row(result_512))
    print()

    print("-" * 70)
    print("【逐问题命中排名对比】（1 = 第一条就命中）")
    print(f"{'问题':<34} {'256':>6} {'512':>6}")
    for q256, q512 in zip(result_256["per_question"], result_512["per_question"]):
        q = q256["question"][:32]
        print(f"{q:<34} {str(q256['hit_rank']):>6} {str(q512['hit_rank']):>6}")
    print()

    # 分块明细（512）
    print("-" * 70)
    print("【512 分块明细】")
    for c in result_512["chunks"]:
        print(
            f"  - {c.metadata['filename']} "
            f"chunk{c.metadata.get('chunk_index')} "
            f"({len(c.page_content)} 字符): "
            f"{c.page_content[:38].replace(chr(10), ' ')}..."
        )

    # 结论
    print()
    print("=" * 70)
    print("【结论】")
    print(f"分块数: {result_256['chunk_count']} -> {result_512['chunk_count']} "
          f"（{result_512['chunk_count'] - result_256['chunk_count']:+d}）")
    print(f"平均块长: {result_256['avg_chunk_len']:.1f} -> {result_512['avg_chunk_len']:.1f} 字符")
    print(f"Recall@1: {result_256['recall_at_1']:.1%} -> {result_512['recall_at_1']:.1%}")
    print(f"Recall@3: {result_256['recall_at_3']:.1%} -> {result_512['recall_at_3']:.1%}")
    print(f"MRR: {result_256['mrr']:.3f} -> {result_512['mrr']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
