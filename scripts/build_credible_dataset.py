#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

SEED = "tiny-decision-stack-credible-v1-2026-10-04"
OUT_DIR = Path("benchmarks/credible-v1")
BOOLQ_REV = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
ARC_REV = "210d026faf9955653af8916fad021475a3f00453"
BANK_REPO_REV = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANK_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANK_REPO_REV}/banking_data"
BANK_SOURCE_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def rank_key(stable_id: str, salt: str = "") -> str:
    return hashlib.sha256(f"{SEED}|{salt}|{stable_id}".encode()).hexdigest()


def canonical_line(item: dict) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(canonical_line(row) for row in rows), encoding="utf-8")


def select_hash(rows: list[dict], n: int, *, salt: str) -> list[dict]:
    if n > len(rows):
        raise ValueError(f"requested {n} from only {len(rows)} rows")
    return sorted(rows, key=lambda r: rank_key(r["id"], salt))[:n]


def boolq_rows(split, split_name: str) -> list[dict]:
    rows = []
    for idx, item in enumerate(split):
        label = "yes" if bool(item["answer"]) else "no"
        rows.append(
            {
                "id": f"boolq:{split_name}:{idx}",
                "task": "boolq",
                "source_split": split_name,
                "state": item["passage"].strip(),
                "question": item["question"].strip(),
                "options": {"no": "No", "yes": "Yes"},
                "label": label,
                "preprocess": "auto",
                "clarify_on_low_confidence": True,
            }
        )
    return rows


def arc_rows(split, split_name: str) -> list[dict]:
    rows = []
    for idx, item in enumerate(split):
        labels = [str(x) for x in item["choices"]["label"]]
        texts = [str(x) for x in item["choices"]["text"]]
        options = dict(zip(labels, texts))
        answer = str(item["answerKey"])
        if answer not in options:
            # Some ARC variants encode numeric keys; normalize if possible.
            if answer.isdigit() and int(answer) - 1 < len(labels):
                answer = labels[int(answer) - 1]
            else:
                raise ValueError(f"ARC answer {answer!r} not in options for {item['id']}")
        rows.append(
            {
                "id": f"arc_challenge:{split_name}:{item['id']}",
                "task": "arc_challenge",
                "source_split": split_name,
                "state": item["question"].strip(),
                "question": "Which answer choice is correct?",
                "options": options,
                "label": answer,
                "preprocess": "auto",
                "clarify_on_low_confidence": True,
            }
        )
    return rows


def download_csv(name: str) -> list[dict[str, str]]:
    url = f"{BANK_BASE}/{name}"
    raw = urllib.request.urlopen(url, timeout=60).read()
    got = digest_bytes(raw)
    expected = BANK_SOURCE_SHA256[name]
    if got != expected:
        raise RuntimeError(f"Banking77 {name} SHA256 drift: {got} != {expected}")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def banking_rows(records: list[dict[str, str]], split_name: str, categories: list[str]) -> list[dict]:
    options = {cat: cat.replace("_", " ") for cat in categories}
    rows = []
    for idx, item in enumerate(records):
        label = item["category"].strip()
        rows.append(
            {
                "id": f"banking77:{split_name}:{idx}",
                "task": "banking77",
                "source_split": split_name,
                "state": item["text"].strip(),
                "question": "Which banking customer-service intent best matches this request?",
                "options": options,
                "label": label,
                "preprocess": "auto",
                "clarify_on_low_confidence": True,
            }
        )
    return rows


def select_banking_calibration(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    selected = []
    for label in sorted(groups):
        selected.extend(sorted(groups[label], key=lambda r: rank_key(r["id"], "bank-cal"))[:2])
    if len(selected) != 154:
        raise ValueError(f"expected 154 Banking77 calibration rows, got {len(selected)}")
    return sorted(selected, key=lambda r: r["id"])


def select_banking_holdout(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    chosen: list[dict] = []
    leftovers: dict[str, list[dict]] = {}
    for label in sorted(groups):
        ordered = sorted(groups[label], key=lambda r: rank_key(r["id"], "bank-holdout"))
        chosen.extend(ordered[:2])
        leftovers[label] = ordered[2:]
    # 154 base cases cover all 77 intents. Add one extra from 46 deterministically chosen intents.
    extra_labels = sorted(leftovers, key=lambda label: rank_key(label, "bank-extra-label"))[:46]
    for label in extra_labels:
        chosen.append(leftovers[label][0])
    if len(chosen) != 200:
        raise ValueError(f"expected 200 Banking77 holdout rows, got {len(chosen)}")
    return sorted(chosen, key=lambda r: r["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the frozen credible-v1 benchmark suite")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the regeneration-only dependency: uv run --with 'datasets>=4,<5' python scripts/build_credible_dataset.py") from exc

    boolq = load_dataset("google/boolq", revision=BOOLQ_REV)
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", revision=ARC_REV)
    bank_train_raw = download_csv("train.csv")
    bank_test_raw = download_csv("test.csv")
    categories = sorted({r["category"].strip() for r in bank_train_raw + bank_test_raw})
    if len(categories) != 77:
        raise ValueError(f"expected 77 Banking77 intents, got {len(categories)}")

    bq_cal = select_hash(boolq_rows(boolq["train"], "train"), 100, salt="boolq-cal")
    bq_hold = select_hash(boolq_rows(boolq["validation"], "validation"), 200, salt="boolq-holdout")
    arc_cal = select_hash(arc_rows(arc["train"], "train"), 100, salt="arc-cal")
    arc_hold = select_hash(arc_rows(arc["test"], "test"), 200, salt="arc-holdout")
    bank_cal_all = banking_rows(bank_train_raw, "train", categories)
    bank_hold_all = banking_rows(bank_test_raw, "test", categories)
    bank_cal = select_banking_calibration(bank_cal_all)
    bank_hold = select_banking_holdout(bank_hold_all)

    calibration = sorted(bq_cal + arc_cal + bank_cal, key=lambda r: (r["task"], r["id"]))
    holdout = sorted(bq_hold + arc_hold + bank_hold, key=lambda r: (r["task"], r["id"]))
    if len(calibration) != 354 or len(holdout) != 600:
        raise AssertionError((len(calibration), len(holdout)))
    overlap = {r["id"] for r in calibration} & {r["id"] for r in holdout}
    if overlap:
        raise AssertionError(f"calibration/holdout overlap: {sorted(overlap)[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = args.out_dir / "calibration.jsonl"
    hold_path = args.out_dir / "holdout.jsonl"
    write_jsonl(cal_path, calibration)
    write_jsonl(hold_path, holdout)

    def counts(rows: list[dict]) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for row in rows:
            out[row["task"]] += 1
        return dict(sorted(out.items()))

    manifest = {
        "suite": "credible-v1",
        "selection_seed": SEED,
        "protocol": {
            "calibration_role": "threshold selection only",
            "holdout_role": "final unbiased evaluation only",
            "target_holdout_count": 600,
            "calibration_count": len(calibration),
            "holdout_count": len(holdout),
            "calibration_by_task": counts(calibration),
            "holdout_by_task": counts(holdout),
            "banking77_holdout": "stratified: 2 per intent + 1 extra from 46 deterministically ranked intents",
        },
        "sources": {
            "boolq": {
                "dataset": "google/boolq",
                "revision": BOOLQ_REV,
                "license": "CC BY-SA 3.0",
                "calibration_source_split": "train",
                "holdout_source_split": "validation",
                "upstream": "https://huggingface.co/datasets/google/boolq",
            },
            "arc_challenge": {
                "dataset": "allenai/ai2_arc / ARC-Challenge",
                "revision": ARC_REV,
                "license": "CC BY-SA 4.0",
                "calibration_source_split": "train",
                "holdout_source_split": "test",
                "upstream": "https://huggingface.co/datasets/allenai/ai2_arc",
            },
            "banking77": {
                "dataset": "BANKING77",
                "revision": BANK_REPO_REV,
                "license": "CC BY 4.0",
                "calibration_source_split": "train",
                "holdout_source_split": "test",
                "upstream": "https://github.com/PolyAI-LDN/task-specific-datasets",
                "source_sha256": BANK_SOURCE_SHA256,
            },
        },
        "files": {
            "calibration.jsonl": {"sha256": digest_file(cal_path), "count": len(calibration)},
            "holdout.jsonl": {"sha256": digest_file(hold_path), "count": len(holdout)},
        },
        "limitations": [
            "Threshold selection is disjoint from holdout evaluation, but these are public benchmarks and may have appeared in model pretraining.",
            "The suite measures zero-shot decision/classification behavior for these tasks; it does not prove universal decision quality.",
            "Banking77 uses all 77 intent names as candidate options, so it is materially harder than binary/four-way tasks.",
        ],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["files"], indent=2))
    print("calibration_by_task", counts(calibration))
    print("holdout_by_task", counts(holdout))


if __name__ == "__main__":
    main()
