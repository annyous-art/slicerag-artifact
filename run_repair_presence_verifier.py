#!/usr/bin/env python3
"""Run a repair-presence verifier on patch-aware hard negative candidates.

This verifier is intentionally narrower than a generic vulnerability verifier:
it receives historical vulnerable->fixed repair evidence and checks whether the
target function already contains the repair obligation shown by that evidence.

Verdict semantics:
- NO: the patch repair is present and adequately guards/replaces the vulnerable
  behavior; reject the first-stage YES.
- YES: the repair is absent, incomplete, or the target still preserves the
  vulnerable-side behavior; keep the first-stage YES.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluate_prompt_outputs import parse_prediction


SYS_REPAIR_VERIFIER = """You are a repair-presence verifier for vulnerability detection.
Your job is not to judge whether code looks generally risky. Your job is to compare a target function against historical vulnerable-to-fixed repair evidence.

Procedure:
1. Extract the concrete repair obligation from the historical evidence: added guard, bound/shape check, API replacement, state reset, lifetime copy, NULL/error handling, or other specific repair.
2. Locate the corresponding logic in the target function.
3. Decide whether the target already contains the repair and whether it protects the same risky operation/path.

Important rules:
- Do not answer YES merely because the target contains parsers, pointer arithmetic, memory APIs, length/index operations, error handling, or other risky-looking code.
- Answer NO only when the target contains the specific repair/check/replacement shown by the fixed side and it appears to guard the same kind of risky operation/path.
- Answer YES only when the target lacks the repair, the repair is incomplete/too late/checks the wrong value, or the vulnerable-side behavior is still present.
- If the historical evidence is irrelevant, from a different behavior pattern, or cannot be mapped to the target, say repair_present=false and keep YES. Do not use irrelevant evidence to reject a first-stage YES.
- Treat retrieval scores as weak hints only. A high score is not enough for NO unless the repair obligation is behaviorally mapped to the target.
- If verdict and repair_present conflict, prefer this invariant: repair_present=true means verdict must be NO; repair_present=false means verdict should be YES unless the evidence is irrelevant.

Return JSON only."""


USER_TEMPLATE = """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because the patch repair is already present.

Metadata:
- project: {project}
- CWE: {cwe}
- repair signal hint: {repair_signal}
- semantic category hint: {primary_category}

Historical vulnerable-to-fixed repair evidence:
```text
{repair_evidence}
```

Target function:
```c
{target_func}
```

Return exactly this JSON object:
{{
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "added_guard|bounds_or_shape_check|api_replacement|state_or_lifetime_repair|error_handling|unknown",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target code condition/API/state change that supports the verdict",
  "reason": "one concise sentence"
}}
"""


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(YES|NO)"', re.IGNORECASE)
REPAIR_PRESENT_RE = re.compile(r'"repair_present"\s*:\s*(true|false)', re.IGNORECASE)


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def truncate_middle(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n/* ... middle omitted for repair-presence verifier ... */\n\n" + text[-tail:]


def extract_target_func_from_prompt(prompt: str) -> str:
    marker = "Target function:\n```c\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = prompt.find("\n```", start)
    if end < 0:
        return prompt[start:]
    return prompt[start:end]


def extract_diff_from_prompt(prompt: str) -> str:
    marker = "Historical paired diff:\n```diff\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = prompt.find("\n```", start)
    if end < 0:
        return prompt[start:]
    return prompt[start:end]


def build_messages(record: dict, max_code_chars: int, max_diff_chars: int) -> list[dict[str, str]]:
    existing_prompt = safe_str(record.get("prompt"))
    repair_evidence = safe_str(
        record.get("repair_evidence")
        or record.get("paired_diff")
        or record.get("diff_excerpt")
        or extract_diff_from_prompt(existing_prompt)
    )
    target_func = safe_str(record.get("target_func") or record.get("func") or extract_target_func_from_prompt(existing_prompt))
    user = USER_TEMPLATE.format(
        project=record.get("project", ""),
        cwe=record.get("cwe", ""),
        repair_signal=record.get("repair_signal", ""),
        primary_category=record.get("primary_category", ""),
        repair_evidence=truncate_middle(repair_evidence, max_diff_chars),
        target_func=truncate_middle(target_func, max_code_chars),
    )
    return [
        {"role": "system", "content": SYS_REPAIR_VERIFIER},
        {"role": "user", "content": user},
    ]


def parse_json_object(response: Any) -> dict:
    text = safe_str(response).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = JSON_OBJECT_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def parse_verdict(response: Any):
    parsed = parse_json_object(response)
    verdict = safe_str(parsed.get("verdict")).upper()
    repair_present_value = parsed.get("repair_present")
    if isinstance(repair_present_value, bool):
        return 0 if repair_present_value else 1
    if isinstance(repair_present_value, str):
        if repair_present_value.lower() == "true":
            return 0
        if repair_present_value.lower() == "false":
            return 1
    if verdict == "YES":
        return 1
    if verdict == "NO":
        return 0
    text = safe_str(response)
    match = VERDICT_RE.search(text)
    if match:
        return 1 if match.group(1).upper() == "YES" else 0
    pred = parse_prediction(text)
    if pred in (0, 1):
        return pred
    repair_present = parse_repair_present(response)
    if repair_present is True:
        return 0
    if repair_present is False:
        return 1
    return None


def parse_repair_present(response: Any):
    parsed = parse_json_object(response)
    value = parsed.get("repair_present")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    match = REPAIR_PRESENT_RE.search(safe_str(response))
    if match:
        return match.group(1).lower() == "true"
    return None


def parsed_field(response: Any, field: str) -> str:
    parsed = parse_json_object(response)
    value = parsed.get(field, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return safe_str(value)


def response_inconsistent(response: Any) -> int:
    parsed = parse_json_object(response)
    verdict = safe_str(parsed.get("verdict")).upper()
    repair_present = parsed.get("repair_present")
    if verdict not in {"YES", "NO"} or not isinstance(repair_present, bool):
        return 0
    return int((verdict == "YES" and repair_present) or (verdict == "NO" and not repair_present))


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


def binary_metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = unknown = 0
    for row in rows:
        target = safe_int(row.get("target"))
        pred = safe_int(row.get("repair_verifier_pred"))
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


def repair_metrics(rows: list[dict]) -> dict:
    known = [row for row in rows if row.get("repair_verifier_pred") in (0, 1)]
    no_count = sum(1 for row in known if row.get("repair_verifier_pred") == 0)
    yes_count = sum(1 for row in known if row.get("repair_verifier_pred") == 1)
    repair_present_known = [row for row in rows if row.get("repair_present") in (True, False)]
    repair_present_count = sum(1 for row in repair_present_known if row.get("repair_present") is True)
    return {
        "known_verdicts": len(known),
        "override_to_no_count": no_count,
        "keep_yes_count": yes_count,
        "override_to_no_rate": no_count / len(known) if known else 0.0,
        "repair_present_known": len(repair_present_known),
        "repair_present_count": repair_present_count,
        "repair_present_rate": repair_present_count / len(repair_present_known) if repair_present_known else 0.0,
    }


def grouped_metrics(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[safe_str(row.get(key) or "unknown")].append(row)
    return {
        name: {**binary_metrics(group_rows), **repair_metrics(group_rows)}
        for name, group_rows in sorted(groups.items())
    }


def summarize(rows: list[dict]) -> dict:
    return {
        **binary_metrics(rows),
        **repair_metrics(rows),
        "category_metrics": grouped_metrics(rows, "primary_category"),
        "repair_signal_metrics": grouped_metrics(rows, "repair_signal"),
        "repair_type_counts": dict(Counter(safe_str(row.get("repair_type") or "unknown") for row in rows).most_common()),
        "confidence_counts": dict(Counter(safe_str(row.get("confidence") or "unknown") for row in rows).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair-presence verifier.")
    parser.add_argument("--input", default="outdir/patch_error_modes/target0_hard_negative_verifier.jsonl")
    parser.add_argument("--output", default="outdir/repair_presence_verifier/repair_presence_outputs.jsonl")
    parser.add_argument("--summary-json", default="outdir/repair_presence_verifier/repair_presence_summary.json")
    parser.add_argument("--model", default="glm-5.1", choices=["gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"])
    parser.add_argument("--prompt_strategy", default="std_cls", choices=["std_cls", "cot"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_gen_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-code-chars", type=int, default=30000)
    parser.add_argument("--max-diff-chars", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input), args.limit)
    prompts = []
    for record in records:
        row = dict(record)
        row["messages"] = build_messages(row, args.max_code_chars, args.max_diff_chars)
        prompts.append(row)

    if args.dry_run:
        out_rows = []
        for row in prompts:
            out = dict(row)
            out["response"] = "[DRY_RUN]"
            out["repair_verifier_pred"] = None
            out["repair_present"] = None
            out_rows.append(out)
        write_jsonl(Path(args.output), out_rows)
        print(f"Wrote dry-run prompts: {args.output}")
        return

    from model_api_clients import get_openai_chat, normalize_usage

    def worker(record: dict) -> dict:
        try:
            response, usage, trunc_messages, reasoning = get_openai_chat(
                {"messages": record["messages"]},
                args.model,
                args.prompt_strategy,
                args.temperature,
                args.max_gen_length,
                args.seed,
            )
        except Exception as exc:
            response, usage, trunc_messages, reasoning = f"[ERROR] {exc}", None, record["messages"], None

        try:
            usage_row = normalize_usage(usage, args.model) if usage is not None else {}
        except Exception:
            usage_row = {}

        out = dict(record)
        out["messages"] = trunc_messages
        out["reasoning"] = reasoning
        out["response"] = response
        out["usage"] = usage_row
        out["repair_verifier_pred"] = parse_verdict(response)
        out["repair_present"] = parse_repair_present(response)
        out["repair_type"] = parsed_field(response, "repair_type")
        out["confidence"] = parsed_field(response, "confidence")
        out["evidence"] = parsed_field(response, "evidence")
        out["reason"] = parsed_field(response, "reason")
        out["repair_verifier_inconsistent"] = response_inconsistent(response)
        target = safe_int(out.get("target"))
        pred = out.get("repair_verifier_pred")
        out["repair_verifier_correct"] = int(pred == target) if pred in (0, 1) and target in (0, 1) else ""
        out["prompt_tokens"] = usage_row.get("prompt_tokens", 0)
        out["completion_tokens"] = usage_row.get("completion_tokens", 0)
        out["reasoning_tokens"] = usage_row.get("reasoning_tokens", 0)
        out["total_tokens"] = usage_row.get("total_tokens", 0)
        return out

    output_rows = []
    output_path = Path(args.output)
    max_workers = max(1, args.num_threads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, record) for record in prompts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Repair verifier"):
            output_rows.append(future.result())
            write_jsonl(output_path, output_rows)

    summary = summarize(output_rows)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote outputs: {output_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
