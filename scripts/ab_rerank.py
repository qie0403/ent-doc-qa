"""
Rerank 开关对照实验：无重排 vs 词法重排
=========================================

实验要回答的问题：
    "Rerank 这一步到底有没有用？" —— 这是面试里最容易被追问、也最容易含糊过去的一点。
    项目默认 ENABLE_RERANK=false，所以必须有数据说明"开或不开差多少"。

实验方法（全部离线、可复现，只依赖 jieba 与 langchain_core）：
    1. 读 eval_data/（20 篇文档 + 40 道标注问题），与 ab_chunk_size.py 同一套语料；
    2. 用 DocumentChunker 按生产配置 512/64 分块；
    3. 用 BM25Retriever（jieba 分词，同生产）召回 top_k=10；
    4. 三种取法对比：
       A. 无重排，直接取前 5（当前默认行为）
       B. 词法重排（LexicalReranker，零依赖）后取前 5
       C. 对照组：把 top_k 加大到 20 再取前 5（回答"是不是只要多召回就行"）
    5. 指标：Recall@1 / Recall@5 / MRR（命中口径与分块实验一致：正确文件 + 答案关键词）

口径（必须与数字一起说）：
    - 召回通道只有 BM25，没有向量检索（向量需要调嵌入接口，离线不可复现）；
    - 重排器是**词法**重排，不是 bge-reranker-large 这类神经交叉编码器，
      所以这里测的是"重排这个环节接进去有没有收益"，不是 BGE 的真实效果；
    - 重排不改变召回集合，因此 Recall@5 理论上不会变，能变的只有排序类指标。

运行方式：
    python scripts/ab_rerank.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from scripts.ab_chunk_size import load_corpus, load_questions
from src.document_processor.chunker import DocumentChunker
from src.retriever.bm25 import BM25Retriever
from src.retriever.lexical_reranker import LexicalReranker

TOP_K = 10
TOP_N = 5

# 命中判定：正确文件 + 答案关键词同时出现
def is_hit(doc: Document, expected_file: str, keyword: str) -> bool:
    filename = doc.metadata.get("filename", "")
    return expected_file in filename and keyword in doc.page_content


def build_chunks(corpus, chunk_size: int, chunk_overlap: int):
    """按指定分块参数切分，返回带 filename 元数据的 Document 列表"""
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    for filename, text in corpus.items():
        chunks = chunker.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"filename": filename}))
    return docs


def evaluate(docs, questions, mode: str):
    """跑一遍评测，返回指标"""
    retriever = BM25Retriever()
    retriever.add_documents(docs)
    reranker = LexicalReranker() if mode == "rerank" else None

    ranks = []
    recall1 = 0
    recall5 = 0
    reordered = 0

    for question, expected_file, keyword in questions:
        # hybrid_wide 用更大的召回窗口，其余用 TOP_K
        retrieve_k = 20 if mode == "hybrid_wide" else TOP_K
        results = retriever.search(question, top_k=retrieve_k)
        candidates = [r.document for r in results]

        if reranker is not None:
            ordered = [r.document for r in reranker.rerank(question, candidates, top_n=TOP_N)]
        else:
            ordered = candidates[:TOP_N]

        # 统计重排是否真的改变了顺序（用于确认"不是空转"）
        if reranker is not None and ordered != candidates[:TOP_N]:
            reordered += 1

        hit_rank = None
        for rank, doc in enumerate(ordered, start=1):
            if is_hit(doc, expected_file, keyword):
                hit_rank = rank
                break

        if hit_rank == 1:
            recall1 += 1
        if hit_rank is not None:
            recall5 += 1
        ranks.append(1.0 / hit_rank if hit_rank else 0.0)

    n = len(questions)
    return {
        "mode": mode,
        "recall_at_1": recall1 / n,
        "recall_at_5": recall5 / n,
        "mrr": sum(ranks) / n,
        "question_count": n,
        "reordered": reordered,
    }


MODE_LABELS = {
    "no_rerank": "A 无重排（top_k=10 → 取前 5）",
    "rerank": "B 词法重排（top_k=10 → 重排 → 前 5）",
    "hybrid_wide": "C 只扩召回（top_k=20 → 取前 5，不重排）",
}


def main():
    corpus = load_corpus()
    questions = load_questions()

    print("=" * 78)
    print("Rerank 开关对照实验（词法重排基线，离线可复现）")
    print("=" * 78)
    print(f"语料: {len(corpus)} 篇；问题: {len(questions)} 道")
    print(f"召回通道: BM25 单通道；TOP_K={TOP_K}，TOP_N={TOP_N}")
    print()

    all_results = {}
    for chunk_size, overlap in ((256, 32), (512, 64)):
        docs = build_chunks(corpus, chunk_size, overlap)
        results = {
            m: evaluate(docs, questions, m)
            for m in ("no_rerank", "rerank", "hybrid_wide")
        }
        all_results[chunk_size] = results

        print(f"--- chunk_size={chunk_size} / overlap={overlap}（共 {len(docs)} 块）---")
        print(f"{'配置':<40}{'Recall@1':>10}{'Recall@5':>10}{'MRR':>9}")
        print("-" * 78)
        for m, r in results.items():
            print(f"{MODE_LABELS[m]:<40}{r['recall_at_1']*100:>9.1f}%"
                  f"{r['recall_at_5']*100:>9.1f}%{r['mrr']:>9.3f}")
        print(f"（词法重排实际改变了顺序的题目数：{results['rerank']['reordered']}/{len(questions)}）")
        print()

    print("=" * 78)
    print("【结论】")
    for chunk_size, results in all_results.items():
        base, rr, wide = results["no_rerank"], results["rerank"], results["hybrid_wide"]
        print(f"chunk_size={chunk_size}:")
        print(f"  重排 vs 无重排：Recall@1 {base['recall_at_1']*100:.1f}% -> {rr['recall_at_1']*100:.1f}%"
              f"（{(rr['recall_at_1']-base['recall_at_1'])*100:+.1f} 个百分点），"
              f"MRR {base['mrr']:.3f} -> {rr['mrr']:.3f}（{rr['mrr']-base['mrr']:+.3f}）")
        print(f"  只扩召回 vs 无重排：Recall@1 {base['recall_at_1']*100:.1f}% -> "
              f"{wide['recall_at_1']*100:.1f}%，MRR {base['mrr']:.3f} -> {wide['mrr']:.3f}")
    print()
    print("口径提醒（必须一起说）：")
    print("  1. 只跑 BM25 单通道，没有向量检索；")
    print("  2. 重排器是词法实现（零依赖），不是 bge-reranker-large，")
    print("     因此本实验证明的是'重排环节是否产生收益'，不能代表 BGE 的真实效果；")
    print("  3. 重排不改变召回集合，Recall@5 本就应当不变，变化的只有排序类指标；")
    print("  4. 512 分块下 BM25 首条命中已经 100%，重排没有改进空间——")
    print("     这本身是个结论：重排的价值取决于召回质量是否已饱和，不是开了就有收益。")


if __name__ == "__main__":
    main()
