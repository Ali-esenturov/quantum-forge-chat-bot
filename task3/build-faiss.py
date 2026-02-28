import os
import re
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


DATA_DIR = "../task2/knowledge-base"
INDEX_PATH = "./nbd.index"
META_PATH = "./nbd_meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_CHARS = 1400
OVERLAP_CHARS = 200


def chunk_text(text: str, chunk_chars: int, overlap_chars: int):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def parse_metadata(raw: str, filename: str):
    doc_id = None
    m = re.search(r"doc_id\s*:\s*([A-Z0-9\-]+)", raw, re.IGNORECASE)
    if m:
        doc_id = m.group(1).strip()

    if not doc_id:
        doc_id = os.path.splitext(os.path.basename(filename))[0]

    def find_field(field):
        mm = re.search(rf"{field}\s*:\s*(.+)", raw, re.IGNORECASE)
        return mm.group(1).strip() if mm else None

    doc_type = find_field("type") or "UNKNOWN"
    date = find_field("date")
    title = find_field("title")

    canaries = re.findall(r"CANARY\s*:\s*([A-Z0-9\-]+)", raw, re.IGNORECASE)

    return {
        "doc_id": doc_id,
        "type": doc_type,
        "date": date,
        "title": title,
        "canaries": list(dict.fromkeys([c.upper() for c in canaries])),
    }


def main():
    model = SentenceTransformer(MODEL_NAME)

    all_texts = []
    all_meta = []

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")]
    files.sort()

    chunk_counter = 0

    for f in files:
        path = os.path.join(DATA_DIR, f)
        with open(path, "r", encoding="utf-8") as fp:
            raw = fp.read()

        meta_doc = parse_metadata(raw, f)

        chunks = chunk_text(raw, CHUNK_CHARS, OVERLAP_CHARS)

        for i, (s, e, chunk) in enumerate(chunks, start=1):
            chunk_id = f"{meta_doc['doc_id']}-CH{i}"
            meta = {
                **meta_doc,
                "chunk_id": chunk_id,
                "source_file": f,
                "char_start": s,
                "char_end": e,
            }
            all_texts.append(chunk)
            all_meta.append(meta)
            chunk_counter += 1

    print(f"Loaded {len(files)} files, produced {chunk_counter} chunks")

    embeddings = model.encode(all_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as fp:
        for m in all_meta:
            fp.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved meta  to {META_PATH}")


if __name__ == "__main__":
    main()