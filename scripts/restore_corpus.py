"""
从 ChromaDB 还原已索引语料到 data/documents/（供分块参数重建实验使用）
运行: python scripts/restore_corpus.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db"))
collection = client.get_collection("enterprise_docs")
data = collection.get()

files = {}
for text, meta in zip(data["documents"], data["metadatas"]):
    name = meta.get("filename", "unknown.txt")
    files.setdefault(name, []).append((meta.get("chunk_index", 0), text))

out_dir = PROJECT_ROOT / "data" / "documents"
out_dir.mkdir(parents=True, exist_ok=True)

for name, chunks in files.items():
    chunks.sort(key=lambda x: x[0])
    texts = [t for _, t in chunks]
    full = texts[0]
    for nxt in texts[1:]:
        overlap = 0
        max_check = min(len(full), len(nxt), 200)
        for size in range(max_check, 0, -1):
            if full[-size:] == nxt[:size]:
                overlap = size
                break
        full += nxt[overlap:]
    out_path = out_dir / name
    out_path.write_text(full, encoding="utf-8")
    print(f"已还原: {out_path} ({len(full)} 字符)")
