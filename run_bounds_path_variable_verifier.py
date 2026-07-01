#!/usr/bin/env python3
"""Run a strict path/variable-level verifier for bounds/shape repairs.

This verifier is intentionally narrower than the generic repair-presence
verifier. It only allows a YES->NO override when the retrieved repair evidence
maps to the same checked variable, same bound variable, same risky sink, same or
stronger relation, and a guard that blocks all relevant paths before the sink.
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


SYS_BOUNDS_PATH_VARIABLE = """You are a path- and variable-level verifier for bounds/shape vulnerability repairs.
Your task is narrow: decide whether a first-stage YES should be rejected because the exact bounds/shape repair obligation from historical evidence is already present in the target function.

Do not perform general vulnerability detection. Do not answer NO merely because the target contains some checks such as IsVector, IsMatrix, offset >= size, NULL checks, OP_REQUIRES, or error returns.

You must verify all required conditions:
1. Same checked variable: the target check maps to the same semantic variable as the repair evidence.
2. Same bound variable: the checked variable is compared against the same semantic size/rank/dimension/length/bound.
3. Same risky sink: the check protects the same kind of operation, such as array access, buffer read/write, memcpy/memmove, pointer arithmetic, tensor reshape/gather/scatter, division/modulo, allocation size, or loop bound.
4. Same or stronger relation: the target check is equivalent to or stronger than the repaired relation. A generic check is not enough.
5. Guard dominates sink: the check occurs before the risky sink and the failure branch blocks the path by return, goto error, break, continue, throw, OP_REQUIRES, or equivalent.
6. No unguarded alternate path: there is no obvious path reaching the risky sink without the check.

Decision rule:
- Return verdict NO and repair_present=true only if all six conditions are satisfied.
- Return verdict YES and repair_present=false if any condition is false, unknown, irrelevant, or only partially satisfied.
- If the historical evidence is about a different operation or cannot be mapped to the target, return YES.

Return JSON only."""


USER_TEMPLATE = """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because a bounds/shape repair is already present at the same variable and path level.

Metadata:
- project: {project}
- CWE: {cwe}
- trigger repair signal: {repair_signal}
- top retrieved pair: {top_pair_id}
- top pair project: {top_pair_project}
- top pair CWE: {top_pair_cwe}
- repair token overlap: {token_overlap}
- repair call overlap: {call_overlap}

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
  "repair_obligation": {{
    "checked_variable": "",
    "bound_variable": "",
    "relation": "",
    "sink_operation": "",
    "failure_action": ""
  }},
  "target_mapping": {{
    "mapped_checked_variable": "",
    "mapped_bound_variable": "",
    "target_guard": "",
    "target_sink": "",
    "same_checked_variable": true or false,
    "same_bound_variable": true or false,
    "same_sink_operation": true or false,
    "same_relation_or_stronger": true or false,
    "guard_before_sink": true or false,
    "failure_blocks_path": true or false,
    "unguarded_alternate_path": true or false
  }},
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "bounds_or_shape_check",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target guard and sink",
  "reason": "one concise sentence"
}}
"""


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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
    return text[:head] + "\n\n/* ... middle omitted for bounds path verifier ... */\n\n" + text[-tail:]


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


def nested(parsed: dict, key: str) -> dict:
    value = parsed.get(key)
    return value if isinstance(value, dict) else {}


def bool_field(mapping: dict, field: str) -> bool:
    value = mapping.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def strict_repair_present(parsed: dict) -> bool:
    mapping = nested(parsed, "target_mapping")
    required_true = [
        "same_checked_variable",
        "same_bound_variable",
        "same_sink_operation",
        "same_relation_or_stronger",
        "guard_before_sink",
        "failure_blocks_path",
    ]
    return all(bool_field(mapping, field) for field in required_true) and not bool_field(mapping, "unguarded_alternate_path")


def has_arithmetic_relation(text: str) -> bool:
    text = safe_str(text)
    return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*[+\-]\s*(?:\d+|[A-Za-z_][A-Za-z0-9_]*))", text))


def relation_arithmetic_compatible(parsed: dict) -> bool:
    """Reject loose LLM equivalence for arithmetic bounds.

    If the historical obligation involves arithmetic such as `offset - 2 > sz`
    or `off + 32 >= size`, a plain target guard like `offset >= sz` is not
    accepted as equivalent by assertion alone.
    """
    obligation = nested(parsed, "repair_obligation")
    mapping = nested(parsed, "target_mapping")
    repair_text = " ".join(
        [
            safe_str(obligation.get("checked_variable")),
            safe_str(obligation.get("relation")),
            safe_str(obligation.get("sink_operation")),
        ]
    )
    if not has_arithmetic_relation(repair_text):
        return True
    target_text = " ".join(
        [
            safe_str(mapping.get("target_guard")),
            safe_str(mapping.get("target_sink")),
            safe_str(mapping.get("mapped_checked_variable")),
        ]
    )
    return has_arithmetic_relation(target_text)


def parse_verdict(response: Any):
    parsed = parse_json_object(response)
    if strict_repair_present(parsed) and relation_arithmetic_compatible(parsed):
        return 0
    repair_present = parsed.get("repair_present")
    if repair_present is True:
        # Be conservative: the structured path/variable checks override a loose boolean.
        return 0 if strict_repair_present(parsed) and relation_arithmetic_compatible(parsed) else 1
    if repair_present is False:
        return 1
    verdict = safe_str(parsed.get("verdict")).upper()
    if verdict == "NO" and strict_repair_present(parsed) and relation_arithmetic_compatible(parsed):
        return 0
    if verdict in {"YES", "NO"}:
        return 1
    return None


def parse_repair_present(response: Any) -> bool | None:
    parsed = parse_json_object(response)
    if strict_repair_present(parsed) and relation_arithmetic_compatible(parsed):
        return True
    if parsed:
        return False
    return None


def parsed_field(response: Any, field: str) -> str:
    parsed = parse_json_object(response)
    value = parsed.get(field, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return safe_str(value)


def parsed_nested_field(response: Any, parent: str, field: str) -> str:
    parsed = parse_json_object(response)
    value = nested(parsed, parent).get(field, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return safe_str(value)


def is_bounds_candidate(record: dict) -> bool:
    text = "|".join(
        [
            safe_str(record.get("top_pair_repair_signal")),
            safe_str(record.get("repair_signal")),
            safe_str(record.get("repair_type")),
        ]
    ).lower()
    return "bounds_or_shape_check" in text


def build_messages(record: dict, max_code_chars: int, max_evidence_chars: int) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        project=record.get("project", ""),
        cwe=record.get("cwe", ""),
        repair_signal=record.get("top_pair_repair_signal", record.get("repair_signal", "")),
        top_pair_id=record.get("top_pair_id", ""),
        top_pair_project=record.get("top_pair_project", ""),
        top_pair_cwe=record.get("top_pair_cwe", ""),
        token_overlap=record.get("top_pair_repair_token_overlap", ""),
        call_overlap=record.get("top_pair_repair_call_overlap", ""),
        repair_evidence=truncate_middle(safe_str(record.get("repair_evidence") or record.get("paired_diff")), max_evidence_chars),
        target_func=truncate_middle(safe_str(record.get("target_func") or record.get("func")), max_code_chars),
    )
    return [
        {"role": "system", "content": SYS_BOUNDS_PATH_VARIABLE},
        {"role": "user", "content": user},
    ]


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


def metrics(rows: list[dict]) -> dict:
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


def grouped_counts(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[safe_str(row.get(key) or "unknown")].append(row)
    return {name: metrics(group_rows) for name, group_rows in sorted(groups.items())}


def summarize(rows: list[dict]) -> dict:
    known = [row for row in rows if row.get("repair_verifier_pred") in (0, 1)]
    override_count = sum(1 for row in known if row.get("repair_verifier_pred") == 0)
    present_count = sum(1 for row in rows if row.get("repair_present") is True)

    def count_value(value: Any) -> str:
        if value is None or value == "":
            return "missing"
        return safe_str(value)

    return {
        **metrics(rows),
        "override_to_no_count": override_count,
        "override_to_no_rate": override_count / len(known) if known else 0.0,
        "repair_present_count": present_count,
        "repair_present_rate": present_count / len(rows) if rows else 0.0,
        "confidence_counts": dict(Counter(safe_str(row.get("confidence") or "unknown") for row in rows).most_common()),
        "target_mapping_counts": {
            field: dict(Counter(count_value(row.get(field)) for row in rows).most_common())
            for field in [
                "same_checked_variable",
                "same_bound_variable",
                "same_sink_operation",
                "same_relation_or_stronger",
                "guard_before_sink",
                "failure_blocks_path",
                "unguarded_alternate_path",
                "relation_arithmetic_compatible",
                "strict_structured_repair_present",
            ]
        },
        "project_metrics": grouped_counts(rows, "project"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict bounds path/variable repair verifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--model",
        default="glm-5.1",
        choices=["gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"],
    )
    parser.add_argument("--prompt_strategy", default="std_cls", choices=["std_cls", "cot"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_gen_length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-code-chars", type=int, default=30000)
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    parser.add_argument("--only-bounds-candidates", action="store_true", default=True)
    parser.add_argument("--all-candidates", dest="only_bounds_candidates", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input), args.limit)
    if args.only_bounds_candidates:
        records = [record for record in records if is_bounds_candidate(record)]

    prompt_rows = []
    for record in records:
        row = dict(record)
        row["messages"] = build_messages(row, args.max_code_chars, args.max_evidence_chars)
        prompt_rows.append(row)

    if args.dry_run:
        out_rows = []
        for row in prompt_rows:
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
        out["repair_type"] = "bounds_or_shape_check"
        out["confidence"] = parsed_field(response, "confidence")
        out["evidence"] = parsed_field(response, "evidence")
        out["reason"] = parsed_field(response, "reason")
        out["checked_variable"] = parsed_nested_field(response, "repair_obligation", "checked_variable")
        out["bound_variable"] = parsed_nested_field(response, "repair_obligation", "bound_variable")
        out["relation"] = parsed_nested_field(response, "repair_obligation", "relation")
        out["sink_operation"] = parsed_nested_field(response, "repair_obligation", "sink_operation")
        out["failure_action"] = parsed_nested_field(response, "repair_obligation", "failure_action")
        parsed = parse_json_object(response)
        mapping = nested(parsed, "target_mapping")
        out["relation_arithmetic_compatible"] = relation_arithmetic_compatible(parsed)
        out["strict_structured_repair_present"] = strict_repair_present(parsed)
        for field in [
            "same_checked_variable",
            "same_bound_variable",
            "same_sink_operation",
            "same_relation_or_stronger",
            "guard_before_sink",
            "failure_blocks_path",
            "unguarded_alternate_path",
        ]:
            out[field] = bool_field(mapping, field)
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
    with ThreadPoolExecutor(max_workers=max(1, args.num_threads)) as executor:
        futures = [executor.submit(worker, record) for record in prompt_rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Bounds path verifier"):
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
