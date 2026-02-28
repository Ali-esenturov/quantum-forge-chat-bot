import os
import re
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "../task2/knowledge-base"
INDEX_PATH = "../task3/nbd.index"
META_PATH = "../task3/nbd_meta.jsonl"
STATE_PATH = "./index_state.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

CHUNK_CHARS = 1400
OVERLAP_CHARS = 200


def file_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
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


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_meta_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def parse_metadata(raw: str, filename: str) -> Dict[str, Any]:
    def find_field(field: str):
        m = re.search(rf"^{field}\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    doc_id = find_field("doc_id")
    if not doc_id:
        doc_id = os.path.splitext(os.path.basename(filename))[0]

    doc_type = find_field("type") or "UNKNOWN"
    date = find_field("date")
    title = find_field("#")

    m_title = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
    title = m_title.group(1).strip() if m_title else None

    canaries = re.findall(r"CANARY\s*:\s*([A-Z0-9\-]+)", raw, flags=re.IGNORECASE)
    canaries = list(dict.fromkeys([c.upper() for c in canaries]))

    return {
        "doc_id": doc_id,
        "type": doc_type,
        "date": date,
        "title": title,
        "canaries": canaries,
    }


def bootstrap_state_without_reindex():
    if os.path.exists(STATE_PATH):
        return False  # no bootstrap needed

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")])
        state = {"files": {}}
        for fname in files:
            path = os.path.join(DATA_DIR, fname)
            sha = file_fingerprint(path)
            state["files"][fname] = {"sha256": sha, "updated_at": datetime.utcnow().isoformat() + "Z"}

        save_json(STATE_PATH, state)
        print(
            "Bootstrapped index_state.json from current files. "
            "Index+meta already existed, so no vectors were added."
        )
        return True

    return False


def main():
    if bootstrap_state_without_reindex():
        return

    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)

    state = load_json(STATE_PATH, default={"files": {}})
    known = state["files"]

    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")])
    to_process = []

    for fname in all_files:
        path = os.path.join(DATA_DIR, fname)
        sha = file_fingerprint(path)

        if fname not in known or known[fname]["sha256"] != sha:
            to_process.append((fname, path, sha))

    if not to_process:
        print("No changes detected. Nothing to update.")
        return

    current_meta_count = load_meta_lines(META_PATH)
    new_meta_records: List[Dict[str, Any]] = []
    new_texts: List[str] = []

    for fname, path, sha in to_process:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        doc_meta = parse_metadata(raw, fname)
        chunks = chunk_text(raw, CHUNK_CHARS, OVERLAP_CHARS)

        for i, (s, e, chunk) in enumerate(chunks, start=1):
            chunk_id = f"{doc_meta['doc_id']}-CH{i}-V{sha[:8]}"

            meta = {
                **doc_meta,
                "chunk_id": chunk_id,
                "source_file": fname,
                "char_start": s,
                "char_end": e,
                "indexed_at": datetime.utcnow().isoformat() + "Z",
                "file_sha256": sha,
            }

            new_texts.append(chunk)
            new_meta_records.append(meta)

        known[fname] = {"sha256": sha, "updated_at": datetime.utcnow().isoformat() + "Z"}

    embs = model.encode(new_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    embs = np.array(embs).astype("float32")
    index.add(embs)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "a", encoding="utf-8") as f:
        for rec in new_meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    save_json(STATE_PATH, state)

    print(f"Updated. Added {len(new_meta_records)} chunks.")
    print(f"FAISS ntotal is now: {index.ntotal}")
    print(f"Meta lines before: {current_meta_count}, after append: {current_meta_count + len(new_meta_records)}")


if __name__ == "__main__":
    main()