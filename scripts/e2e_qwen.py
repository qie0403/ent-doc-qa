# -*- coding: utf-8 -*-
"""换模型后端到端验证：初始化pipeline + 流式问答 + 多轮改写 + 引用格式"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import QAPipeline

p = QAPipeline()
stats = p.get_stats()
print("=== 系统状态 ===")
print("片段数:", stats.total_documents)

# 1) 单轮流式问答（同时覆盖 query embedding 检索 + 推理模型流式）
q1 = "减少大模型幻觉的一种策略是什么？"
print("\n=== 问题1（流式）:", q1)
sid = "e2e_test"
chunks = []
final_resp = None
for item in p.ask_stream(q1, session_id=sid):
    chunk, resp = item
    if chunk:
        chunks.append(chunk)
    if resp is not None:
        final_resp = resp
answer = "".join(chunks)
print("答案长度:", len(answer))
print("答案:", answer[:400].replace("\n", " "))
if final_resp is not None:
    print("来源数:", len(final_resp.sources))
    for s in final_resp.sources:
        print("  -", s.filename, "| 分数:", s.score)
else:
    hist = p.get_history(sid)
    msgs = [m for m in hist.messages if m.get("role") == "assistant"]
    print("来源:", msgs[-1].get("sources") if msgs else None)

# 2) 多轮追问（触发 query 改写）
q2 = "那它具体是怎么实现的？"
print("\n=== 问题2（多轮追问，触发改写）:", q2)
chunks2 = []
for item in p.ask_stream(q2, session_id=sid):
    chunk, resp = item
    if chunk:
        chunks2.append(chunk)
answer2 = "".join(chunks2)
print("答案长度:", len(answer2))
print("答案:", answer2[:400].replace("\n", " "))
print("是否含引用标记[来源N]:", "来源" in answer2)

# 3) 非流式 ask 兜底
q3 = "LangChain 是什么？"
print("\n=== 问题3（非流式）:", q3)
r = p.ask(q3, session_id=sid + "_2")
print("answer 类型:", type(r.get("answer")).__name__ if hasattr(r, "keys") else type(r).__name__, "| 长度:", len(r.answer))
print("来源数:", len(r.sources))
print("检索片段数:", len(r.retrieved_docs))
print("耗时(ms):", r.latency_ms)
print("非流式答案:", r.answer[:200].replace("\n", " "))
