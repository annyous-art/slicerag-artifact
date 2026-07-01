#!/usr/bin/env python3
"""Run a second-stage verifier on SliceRAG disagreement samples."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluate_prompt_outputs import parse_prediction


SYS_INST_VERIFIER_V1_CONSERVATIVE = """You are a careful security verifier for C/C++ vulnerability detection.
Your job is to adjudicate a disagreement between a high-recall first-stage classifier and a more conservative retrieval/ICL method.
Use the target function code as the primary evidence.
Do not classify YES just because the code contains risky APIs, pointer operations, NULL checks, error handling, or complex control flow.
Classify YES only if there is a concrete vulnerability mechanism in the target code, such as an unchecked bounds/path condition, unsafe lifetime/use-after-free, missing NULL check before dereference, integer overflow that affects memory access/allocation, or an error path that can lead to unsafe behavior.
If the risky operation appears guarded by adequate checks in the shown function, answer NO.
"""


USER_TEMPLATE_V1_CONSERVATIVE = """A first-stage classifier predicted YES for the target function, while one or more conservative methods predicted NO.

Previous model predictions:
- zero_shot: {zero_shot_pred_text}
- ICL: {icl_pred_text}
- old_RAG: {old_rag_pred_text}
- positive_weighted_RAG: {positive_weighted_rag_pred_text}
- balanced_grouped_no_label_RAG: {balanced_grouped_no_label_bc_pred_text}

Metadata, for context only:
- project: {project}
- CWE: {cwe}
- function length: {func_char_len} chars, {func_line_count} lines
- common calls: {top_calls}

Target Function Code:
```c
{func}
```

Verifier task:
Decide whether the target function contains a security vulnerability.
Focus on whether the code has a concrete unsafe path, not merely suspicious tokens.

Reply in this exact JSON format:
{{
  "verdict": "YES" or "NO",
  "confidence": "low" or "medium" or "high",
  "reason": "one or two concise sentences explaining the concrete vulnerability path or why the suspected risk is guarded"
}}
"""

SYS_INST_VERIFIER_V2_BALANCED = """You are a balanced security verifier for C/C++ vulnerability detection.
Your job is to re-evaluate a target function where previous models disagreed.
Use the target function code as the primary evidence.

Avoid both common failure modes:
1. Do not answer YES merely because the code contains risky APIs, pointer arithmetic, NULL checks, error handling, or complex control flow.
2. Do not answer NO merely because some checks are present. A vulnerability may still exist if a check is incomplete, too late, checks the wrong quantity, misses an alias/lifetime condition, or relies on an invalid assumption about attacker-controlled input.

Answer YES if the target code has a plausible concrete vulnerability mechanism, including but not limited to:
- out-of-bounds read/write from unchecked or incorrectly checked length/index/offset;
- integer overflow/truncation that affects allocation, copy size, loop bound, or pointer arithmetic;
- NULL/error-pointer dereference on an unchecked path;
- use-after-free, double free, refcount/lifetime misuse, or cleanup error path misuse;
- missing validation of externally controlled structure/shape/header fields before use;
- security policy/authentication/permission checks that are skipped or incomplete.

Answer NO if the suspected risky operation is actually guarded on all relevant paths in the shown code, or if the concern depends only on external assumptions not evidenced by the target function.
Be especially careful with functions from vulnerability datasets: the vulnerable path may be subtle and may involve edge cases, malformed inputs, or error paths.
"""


USER_TEMPLATE_V2_BALANCED = """Previous models disagree on whether the target function is vulnerable.

Prediction summary, for context only:
- zero_shot: {zero_shot_pred_text}
- ICL: {icl_pred_text}
- old_RAG: {old_rag_pred_text}
- positive_weighted_RAG: {positive_weighted_rag_pred_text}
- balanced_grouped_no_label_RAG: {balanced_grouped_no_label_bc_pred_text}

Do not follow any previous prediction blindly. Re-evaluate the target code directly.

Metadata, for context only:
- project: {project}
- CWE: {cwe}
- function length: {func_char_len} chars, {func_line_count} lines
- common calls: {top_calls}

Target Function Code:
```c
{func}
```

Verifier checklist:
- Identify the most security-relevant operation or error/lifetime path.
- Check whether validation occurs before use and covers the same variable/size/object that is later used.
- Check off-by-one conditions such as < vs <=, p > end vs p >= end, and multiplication/addition overflow before allocation or copy.
- Check whether pointer/refcount/resource cleanup remains valid across all branches.
- If the function appears safe, explain which checks guard the suspected risk.

Reply in this exact JSON format:
{{
  "verdict": "YES" or "NO",
  "confidence": "low" or "medium" or "high",
  "reason": "one or two concise sentences explaining the concrete vulnerability path or why the suspected risk is guarded"
}}
"""

USER_TEMPLATE_V3_BLINDED = """Previous models disagree on whether the target function is vulnerable.
The individual model predictions are intentionally hidden. Re-evaluate the target code directly without anchoring to any prior answer.

Metadata, for context only:
- project: {project}
- CWE: {cwe}
- function length: {func_char_len} chars, {func_line_count} lines
- common calls: {top_calls}

Target Function Code:
```c
{func}
```

Verifier checklist:
- Identify the most security-relevant operation or error/lifetime path.
- Check whether validation occurs before use and covers the same variable/size/object that is later used.
- Check off-by-one conditions such as < vs <=, p > end vs p >= end, and multiplication/addition overflow before allocation or copy.
- Check whether pointer/refcount/resource cleanup remains valid across all branches.
- If the function appears safe, explain which checks guard the suspected risk.

Reply in this exact JSON format:
{{
  "verdict": "YES" or "NO",
  "confidence": "low" or "medium" or "high",
  "reason": "one or two concise sentences explaining the concrete vulnerability path or why the suspected risk is guarded"
}}
"""


VERIFIER_PROMPTS = {
    "v1_conservative": (SYS_INST_VERIFIER_V1_CONSERVATIVE, USER_TEMPLATE_V1_CONSERVATIVE),
    "v2_balanced": (SYS_INST_VERIFIER_V2_BALANCED, USER_TEMPLATE_V2_BALANCED),
    "v3_blinded": (SYS_INST_VERIFIER_V2_BALANCED, USER_TEMPLATE_V3_BLINDED),
}


YES_RE = re.compile(r'"verdict"\s*:\s*"YES"', re.IGNORECASE)
NO_RE = re.compile(r'"verdict"\s*:\s*"NO"', re.IGNORECASE)


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def pred_text(value: Any) -> str:
    pred = safe_int(value)
    if pred == 1:
        return "YES"
    if pred == 0:
        return "NO"
    return "UNKNOWN"


def truncate_code_chars(func: str, max_chars: int) -> str:
    func = func or ""
    if max_chars <= 0 or len(func) <= max_chars:
        return func
    head = max_chars // 2
    tail = max_chars - head
    return func[:head] + "\n\n/* ... middle of long function omitted for verifier prompt ... */\n\n" + func[-tail:]


def build_messages(record: dict, max_code_chars: int, include_cwe: bool, verifier_prompt: str) -> list[dict[str, str]]:
    func = truncate_code_chars(record.get("func", ""), max_code_chars)
    cwe = record.get("cwe", "") if include_cwe else "not provided"
    sys_inst, user_template = VERIFIER_PROMPTS[verifier_prompt]
    top_calls = record.get("top_calls", [])
    if isinstance(top_calls, list):
        top_calls_text = ", ".join(str(item) for item in top_calls[:20])
    else:
        top_calls_text = str(top_calls)
    user = user_template.format(
        zero_shot_pred_text=pred_text(record.get("zero_shot_pred")),
        icl_pred_text=pred_text(record.get("icl_pred")),
        old_rag_pred_text=pred_text(record.get("old_rag_pred")),
        positive_weighted_rag_pred_text=pred_text(record.get("positive_weighted_rag_pred")),
        balanced_grouped_no_label_bc_pred_text=pred_text(record.get("balanced_grouped_no_label_bc_pred")),
        project=record.get("project", ""),
        cwe=cwe,
        func_char_len=record.get("func_char_len", ""),
        func_line_count=record.get("func_line_count", ""),
        top_calls=top_calls_text,
        func=func,
    )
    return [
        {"role": "system", "content": sys_inst},
        {"role": "user", "content": user},
    ]


def parse_verdict(response: Any):
    if response is None:
        return None
    text = str(response)
    if YES_RE.search(text):
        return 1
    if NO_RE.search(text):
        return 0
    return parse_prediction(text)


def compute_metrics(rows: list[dict]) -> dict[str, Any]:
    tp = fp = tn = fn = unknown = 0
    for row in rows:
        target = safe_int(row.get("target"))
        pred = safe_int(row.get("verifier_pred"))
        if target not in (0, 1) or pred not in (0, 1):
            unknown += 1
            continue
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 0 and pred == 0:
            tn += 1
        elif target == 1 and pred == 0:
            fn += 1
    known = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_records": len(rows),
        "known_predictions": known,
        "unknown_predictions": unknown,
        "accuracy": (tp + tn) / known if known else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": (tp + fp) / known if known else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_group_metrics(rows: list[dict]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group = str(row.get("verifier_group", "unknown"))
        groups.setdefault(group, []).append(row)
    return {group: compute_metrics(group_rows) for group, group_rows in sorted(groups.items())}


def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run verifier prompts on disagreement samples.")
    parser.add_argument("--input", default="outdir/verifier_sets/verifier_disagreement_sample.jsonl")
    parser.add_argument("--output", default="outdir/verifier_results/verifier_gpt55_sample.jsonl")
    parser.add_argument("--summary-json", default="outdir/verifier_results/verifier_gpt55_sample_metrics.json")
    parser.add_argument("--model", default="gpt-5.5", choices=["gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"])
    parser.add_argument("--prompt_strategy", default="std_cls", choices=["std_cls", "cot"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_gen_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-code-chars", type=int, default=50000)
    parser.add_argument("--include-cwe", action="store_true")
    parser.add_argument("--verifier-prompt", choices=sorted(VERIFIER_PROMPTS), default="v2_balanced")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without calling the model.")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input), args.limit)
    prompts = []
    for record in records:
        messages = build_messages(record, args.max_code_chars, args.include_cwe, args.verifier_prompt)
        prompt_record = dict(record)
        prompt_record["messages"] = messages
        prompt_record["verifier_prompt"] = args.verifier_prompt
        prompts.append(prompt_record)

    if args.dry_run:
        out_rows = []
        for record in prompts:
            row = dict(record)
            row["response"] = "[DRY_RUN]"
            row["verifier_pred"] = ""
            out_rows.append(row)
        write_jsonl(Path(args.output), out_rows)
        print(f"Wrote dry-run prompts: {args.output}")
        return

    from model_api_clients import get_openai_chat, normalize_usage

    def worker(record: dict) -> dict:
        response, usage, trunc_messages, reasoning = get_openai_chat(
            {"messages": record["messages"]},
            args.model,
            args.prompt_strategy,
            args.temperature,
            args.max_gen_length,
            args.seed,
        )
        usage = normalize_usage(usage, args.model)
        row = dict(record)
        row["messages"] = trunc_messages
        row["reasoning"] = reasoning
        row["usage"] = usage
        row["response"] = response if response is not None else "[ERROR]"
        row["verifier_pred"] = parse_verdict(row["response"])
        row["verifier_correct"] = int(row["verifier_pred"] == safe_int(row.get("target"))) if row["verifier_pred"] in (0, 1) else ""
        row["prompt_tokens"] = usage.get("prompt_tokens", 0)
        row["completion_tokens"] = usage.get("completion_tokens", 0)
        row["reasoning_tokens"] = usage.get("reasoning_tokens")
        row["total_tokens"] = usage.get("total_tokens", 0)
        return row

    output_rows = []
    max_workers = max(1, args.num_threads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row in tqdm(executor.map(worker, prompts), total=len(prompts)):
            output_rows.append(row)
            write_jsonl(Path(args.output), output_rows)

    metrics = compute_metrics(output_rows)
    metrics["group_metrics"] = compute_group_metrics(output_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote verifier outputs: {args.output}")
    print(f"Wrote metrics: {args.summary_json}")


if __name__ == "__main__":
    main()
