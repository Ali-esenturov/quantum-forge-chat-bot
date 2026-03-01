import json
import time
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/ask"
GOLD_PATH = "golden-questions.txt"

OUT_JSONL = "logs.jsonl"
OUT_CSV = "logs.csv"

SLEEP_SEC = 0.2


def norm(s: str) -> str:
    return (s or "").strip().lower()


def load_golden(path: str):
    """
    golden_questions.txt format (one per line):
      question<TAB>expected_answer
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError("Each line must be: question<TAB>expected_answer")
            q, exp = line.split("\t", 1)
            items.append((q.strip(), exp.strip()))
    return items


def deterministic_pass(answer: str, expected: str) -> bool:
    """
    Deterministic check:
    - if expected == "I don't know" -> answer must match exactly
    - else -> expected must be a substring of answer (case-insensitive)
    """
    if expected == "I don't know":
        return (answer or "").strip() == "I don't know"
    return norm(expected) in norm(answer)


def main():
    golden = load_golden(GOLD_PATH)

    logs = []
    passed = 0

    for query, expected in golden:
        try:
            r = requests.post(API_URL, json={"question": query}, timeout=60)
        except Exception as e:
            logs.append({
                "query": query,
                "result": "",
                "sources": [],
                "length": 0,
                "status": f"request_error:{type(e).__name__}",
                "expected": expected,
                "pass": False,
            })
            continue

        if r.status_code != 200:
            logs.append({
                "query": query,
                "result": "",
                "sources": [],
                "length": 0,
                "status": f"http_error:{r.status_code}",
                "expected": expected,
                "pass": False,
            })
            continue

        data = r.json()
        answer = data.get("answer", "") or ""
        sources = data.get("sources", []) or []
        max_score = data.get("max_score", None)

        ok = deterministic_pass(answer, expected)
        if ok:
            passed += 1

        if answer.strip() == "I don't know":
            status = "no_knowledge"
        elif ok:
            status = "ok"
        else:
            status = "mismatch"

        logs.append({
            "query": query,
            "result": answer,
            "sources": sources,
            "length": len(answer),
            "status": status,

            "expected": expected,
            "pass": ok,
            "max_score": max_score,
        })

        time.sleep(SLEEP_SEC)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    df = pd.DataFrame(logs)
    df["sources"] = df["sources"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    total = len(golden)
    accuracy = (passed / total) if total else 0.0

    with open("logs_pretty.json", "w", encoding="utf-8") as f:
      json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"Done. Accuracy: {accuracy:.2%} ({passed}/{total})")
    print(f"Wrote: {OUT_JSONL}, {OUT_CSV}")


if __name__ == "__main__":
    main()