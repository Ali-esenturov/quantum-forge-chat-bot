import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "./nbd.index"
META_PATH = "./nbd_meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

def load_meta(path):
    meta = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta

def main():
    print("Loading model/index...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    meta = load_meta(META_PATH)
    print(f"Ready. Chunks in index: {index.ntotal}", flush=True)

    if not sys.stdin.isatty():
        print("stdin is not interactive (likely IDE/run button). Run this script in a Terminal.", flush=True)
        return

    while True:
        try:
            q = input("\nQuery (or 'exit'): ").strip()
        except EOFError:
            print("\nEOF received. Exiting.", flush=True)
            break

        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue

        q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(q_emb, TOP_K)

        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if idx < 0:
                continue
            m = meta[idx]
            print(
                f"\n#{rank} score={score:.3f} doc_id={m.get('doc_id')} chunk_id={m.get('chunk_id')} "
                f"type={m.get('type')} canaries={m.get('canaries')}"
            )
            print(f"source={m.get('source_file')} chars={m.get('char_start')}-{m.get('char_end')}", flush=True)

if __name__ == "__main__":
    main()