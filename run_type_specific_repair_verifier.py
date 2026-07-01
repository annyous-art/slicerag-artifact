#!/usr/bin/env python3
"""Run type-specific repair-presence verifiers.

This script extends the stricter bounds verifier idea to other repair families.
It is still a YES->NO refinement module, not a standalone vulnerability
classifier: a NO means the retrieved patch repair appears to be present in the
target at the same behavior/path level, so the first-stage YES can be rejected.
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


REPAIR_TYPES = {
    "api_replacement",
    "state_or_lifetime_repair",
    "error_handling",
    "null_check",
    "added_guard",
}

TYPE_ALIASES = {
    "state_lifetime": "state_or_lifetime_repair",
    "lifetime": "state_or_lifetime_repair",
    "null_guard": "null_check",
}

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


SYS_TEMPLATE = """You are a type-specific repair-presence verifier for vulnerability detection.
Your job is narrow: decide whether a first-stage YES should be rejected because the specific historical {repair_type} repair is already present in the target function.

Do not perform general vulnerability detection. Do not answer NO because the target merely looks safe or contains some related checks.
Answer NO only when the target contains the same repair obligation from the historical fixed side and that repair protects/replaces the same risky behavior/path.
Answer YES when the repair is absent, incomplete, too late, mapped to the wrong variable/API/object/path, or the historical evidence is irrelevant.

Return JSON only."""


USER_TEMPLATES = {
    "api_replacement": """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because the exact API replacement repair is already present.

Required NO conditions:
1. The historical evidence clearly replaces a vulnerable/unsafe API, call pattern, or operation with a safer API/pattern.
2. The target uses the fixed-side API/pattern for the same semantic operation or dataflow.
3. The vulnerable-side API/pattern is absent, unreachable, or no longer used for that operation.
4. The replacement occurs at the relevant call site/path, not in an unrelated helper branch.

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
    "vulnerable_api_or_pattern": "",
    "fixed_api_or_pattern": "",
    "operation_or_dataflow": "",
    "required_context": ""
  }},
  "target_mapping": {{
    "fixed_api_present": true or false,
    "vulnerable_api_absent_or_unreachable": true or false,
    "same_operation_or_dataflow": true or false,
    "same_callsite_or_path": true or false,
    "replacement_before_risky_behavior": true or false,
    "unreplaced_alternate_path": true or false
  }},
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "api_replacement",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target replacement",
  "reason": "one concise sentence"
}}
""",
    "state_or_lifetime_repair": """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because the exact state/lifetime repair is already present.

Required NO conditions:
1. The historical repair fixes ownership, reference count, release/free order, state flag, object validity, or lifecycle transition.
2. The target applies the same state/lifetime action to the same semantic object/resource.
3. The action occurs before the corresponding use/release/transition that would otherwise be unsafe.
4. No obvious alternate path still uses the object after invalidation or skips the required state/lifetime action.

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
    "object_or_resource": "",
    "lifetime_or_state_action": "",
    "unsafe_event": "",
    "required_ordering": ""
  }},
  "target_mapping": {{
    "same_object_or_resource": true or false,
    "same_lifetime_or_state_action": true or false,
    "action_before_unsafe_event": true or false,
    "required_ordering_preserved": true or false,
    "all_relevant_paths_covered": true or false,
    "unsafe_alternate_path": true or false
  }},
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "state_or_lifetime_repair",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target state/lifetime action",
  "reason": "one concise sentence"
}}
""",
    "error_handling": """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because the exact error-handling repair is already present.

Required NO conditions:
1. The historical repair adds or strengthens a specific error condition, return-code check, failure branch, cleanup path, or exception/abort behavior.
2. The target checks the same semantic failure condition for the same operation.
3. The failure action blocks the vulnerable path by return, goto cleanup/error, throw, break, continue, abort, or equivalent.
4. The check dominates the risky sink; no obvious alternate path reaches the sink without the check.

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
    "checked_condition": "",
    "protected_operation": "",
    "failure_action": "",
    "cleanup_or_return_value": ""
  }},
  "target_mapping": {{
    "same_checked_condition": true or false,
    "same_protected_operation": true or false,
    "failure_action_present": true or false,
    "check_before_operation": true or false,
    "failure_blocks_path": true or false,
    "unchecked_alternate_path": true or false
  }},
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "error_handling",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target error handling",
  "reason": "one concise sentence"
}}
""",
    "null_check": """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because the exact NULL/presence guard repair is already present.

Required NO conditions:
1. The historical repair adds a NULL/presence/validity guard for a specific pointer/object/value.
2. The target checks the same semantic pointer/object/value, not just any nearby pointer.
3. The guard occurs before the dereference/use/sink and failure blocks that path.
4. No obvious alternate dereference/use path skips the guard.

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
    "checked_pointer_or_value": "",
    "guard_condition": "",
    "protected_use_or_deref": "",
    "failure_action": ""
  }},
  "target_mapping": {{
    "same_pointer_or_value": true or false,
    "same_use_or_deref": true or false,
    "guard_before_use": true or false,
    "failure_blocks_path": true or false,
    "unguarded_alternate_use": true or false
  }},
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "null_check",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target NULL/presence guard",
  "reason": "one concise sentence"
}}
""",
    "added_guard": """A first-stage vulnerability detector predicted YES for this target function.
Verify whether that YES should be rejected because the exact non-bounds added guard repair is already present.

Use this verifier only for added guards that are not primarily bounds/shape checks. If the evidence is mainly bounds/shape, return YES.

Required NO conditions:
1. The historical repair adds a specific guard/condition for a concrete risky operation/path.
2. The target contains the same semantic guard over the same variables/state.
3. The guard dominates the same risky operation/path.
4. The failure branch blocks the path.
5. There is no obvious alternate path reaching the operation without the guard.

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
    "guard_condition": "",
    "guarded_operation": "",
    "guarded_variables_or_state": "",
    "failure_action": ""
  }},
  "target_mapping": {{
    "same_guard_condition": true or false,
    "same_guarded_variables_or_state": true or false,
    "same_guarded_operation": true or false,
    "guard_before_operation": true or false,
    "failure_blocks_path": true or false,
    "unguarded_alternate_path": true or false,
    "primarily_bounds_or_shape": true or false
  }},
  "verdict": "YES" or "NO",
  "repair_present": true or false,
  "repair_type": "added_guard",
  "confidence": "low" or "medium" or "high",
  "evidence": "quote or summarize the target guard",
  "reason": "one concise sentence"
}}
""",
}


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_type(value: str) -> str:
    value = value.strip()
    return TYPE_ALIASES.get(value, value)


def split_repair_types(value: Any) -> set[str]:
    out = set()
    for part in safe_str(value).split("|"):
        part = normalize_type(part.strip())
        if part:
            out.add(part)
    return out


def record_repair_types(record: dict) -> set[str]:
    out = set()
    for key in ["top_pair_repair_signal", "repair_signal", "repair_type"]:
        out |= split_repair_types(record.get(key))
    return out


def truncate_middle(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n/* ... middle omitted for type-specific repair verifier ... */\n\n" + text[-tail:]


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


def strict_repair_present(parsed: dict, repair_type: str) -> bool:
    mapping = nested(parsed, "target_mapping")
    if repair_type == "api_replacement":
        return all(
            bool_field(mapping, field)
            for field in [
                "fixed_api_present",
                "vulnerable_api_absent_or_unreachable",
                "same_operation_or_dataflow",
                "same_callsite_or_path",
                "replacement_before_risky_behavior",
            ]
        ) and not bool_field(mapping, "unreplaced_alternate_path")
    if repair_type == "state_or_lifetime_repair":
        return all(
            bool_field(mapping, field)
            for field in [
                "same_object_or_resource",
                "same_lifetime_or_state_action",
                "action_before_unsafe_event",
                "required_ordering_preserved",
                "all_relevant_paths_covered",
            ]
        ) and not bool_field(mapping, "unsafe_alternate_path")
    if repair_type == "error_handling":
        return all(
            bool_field(mapping, field)
            for field in [
                "same_checked_condition",
                "same_protected_operation",
                "failure_action_present",
                "check_before_operation",
                "failure_blocks_path",
            ]
        ) and not bool_field(mapping, "unchecked_alternate_path")
    if repair_type == "null_check":
        return all(
            bool_field(mapping, field)
            for field in [
                "same_pointer_or_value",
                "same_use_or_deref",
                "guard_before_use",
                "failure_blocks_path",
            ]
        ) and not bool_field(mapping, "unguarded_alternate_use")
    if repair_type == "added_guard":
        return all(
            bool_field(mapping, field)
            for field in [
                "same_guard_condition",
                "same_guarded_variables_or_state",
                "same_guarded_operation",
                "guard_before_operation",
                "failure_blocks_path",
            ]
        ) and not bool_field(mapping, "unguarded_alternate_path") and not bool_field(mapping, "primarily_bounds_or_shape")
    return False


def parse_verdict(response: Any, repair_type: str):
    parsed = parse_json_object(response)
    if strict_repair_present(parsed, repair_type):
        return 0
    if parsed:
        return 1
    return None


def parse_repair_present(response: Any, repair_type: str) -> bool | None:
    parsed = parse_json_object(response)
    if strict_repair_present(parsed, repair_type):
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


def build_messages(record: dict, repair_type: str, max_code_chars: int, max_evidence_chars: int) -> list[dict[str, str]]:
    user = USER_TEMPLATES[repair_type].format(
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
        {"role": "system", "content": SYS_TEMPLATE.format(repair_type=repair_type)},
        {"role": "user", "content": user},
    ]


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


def count_value(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    return safe_str(value)


def summarize(rows: list[dict], repair_type: str) -> dict:
    known = [row for row in rows if row.get("repair_verifier_pred") in (0, 1)]
    override_count = sum(1 for row in known if row.get("repair_verifier_pred") == 0)
    present_count = sum(1 for row in rows if row.get("repair_present") is True)
    mapping_fields = sorted(
        {
            field
            for row in rows
            for field in row
            if field.startswith(("same_", "fixed_", "vulnerable_", "replacement_", "unreplaced_", "action_", "required_", "all_", "unsafe_", "failure_", "check_", "unchecked_", "guard_", "unguarded_", "primarily_"))
        }
    )
    return {
        **metrics(rows),
        "repair_type": repair_type,
        "override_to_no_count": override_count,
        "override_to_no_rate": override_count / len(known) if known else 0.0,
        "repair_present_count": present_count,
        "repair_present_rate": present_count / len(rows) if rows else 0.0,
        "confidence_counts": dict(Counter(safe_str(row.get("confidence") or "unknown") for row in rows).most_common()),
        "target_mapping_counts": {
            field: dict(Counter(count_value(row.get(field)) for row in rows).most_common())
            for field in mapping_fields
        },
        "project_metrics": grouped_counts(rows, "project"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a type-specific repair-presence verifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--repair-type", required=True, choices=sorted(REPAIR_TYPES | set(TYPE_ALIASES)))
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
    parser.add_argument("--only-type-candidates", action="store_true", default=True)
    parser.add_argument("--all-candidates", dest="only_type_candidates", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repair_type = normalize_type(args.repair_type)
    records = load_jsonl(Path(args.input), args.limit)
    if args.only_type_candidates:
        records = [record for record in records if repair_type in record_repair_types(record)]

    prompt_rows = []
    for record in records:
        row = dict(record)
        row["verifier_repair_type"] = repair_type
        row["messages"] = build_messages(row, repair_type, args.max_code_chars, args.max_evidence_chars)
        prompt_rows.append(row)

    if args.dry_run:
        out_rows = []
        for row in prompt_rows:
            out = dict(row)
            out["response"] = "[DRY_RUN]"
            out["repair_verifier_pred"] = None
            out["repair_present"] = None
            out["repair_type"] = repair_type
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
        out["repair_verifier_pred"] = parse_verdict(response, repair_type)
        out["repair_present"] = parse_repair_present(response, repair_type)
        out["repair_type"] = repair_type
        out["confidence"] = parsed_field(response, "confidence")
        out["evidence"] = parsed_field(response, "evidence")
        out["reason"] = parsed_field(response, "reason")
        out["strict_structured_repair_present"] = strict_repair_present(parse_json_object(response), repair_type)

        parsed = parse_json_object(response)
        mapping = nested(parsed, "target_mapping")
        for field, value in mapping.items():
            if isinstance(value, bool):
                out[field] = value
            elif isinstance(value, (str, int, float)):
                out[field] = value

        obligation = nested(parsed, "repair_obligation")
        for field, value in obligation.items():
            if isinstance(value, (str, int, float)):
                out[f"obligation_{field}"] = value

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
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{repair_type} verifier"):
            output_rows.append(future.result())
            write_jsonl(output_path, output_rows)

    summary = summarize(output_rows, repair_type)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote outputs: {output_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
