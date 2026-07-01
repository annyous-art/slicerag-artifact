#!/usr/bin/env python3
"""Run bias-controlled prompt variants for vulnerability classification.

These prompts are designed for empirical analysis rather than deployment. They
measure how much YES/NO tendency comes from task framing, metadata, and few-shot
labels when code evidence is removed or perturbed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from Baseline.utils import (
        ONESHOT_ASSISTANT,
        ONESHOT_USER,
        SYS_INST,
        TWOSHOT_ASSISTANT,
        TWOSHOT_USER,
    )
except ModuleNotFoundError:
    from baseline.utils import (
        ONESHOT_ASSISTANT,
        ONESHOT_USER,
        SYS_INST,
        TWOSHOT_ASSISTANT,
        TWOSHOT_USER,
    )
from evaluate_prompt_outputs import parse_prediction


PROMPT_CODE = """Please analyze the following code:
```c
{func}
```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


PROMPT_NO_CODE = """A C/C++ function is being evaluated for security vulnerability detection, but the function code is intentionally hidden for this bias-control experiment.

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


PROMPT_METADATA = """A C/C++ function is being evaluated for security vulnerability detection. The function code is intentionally hidden for this bias-control experiment.

Metadata:
- project: {project}
- file name: {file_name}
- CWE metadata: {cwe}

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""

PROMPT_METADATA_NO_CWE = """A C/C++ function is being evaluated for security vulnerability detection. The function code and CWE/CVE metadata are intentionally hidden for this bias-control experiment.

Metadata:
- project: {project}
- file name: {file_name}

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


PROMPT_CODE_METADATA = """A C/C++ function is being evaluated for security vulnerability detection.

Metadata:
- project: {project}
- file name: {file_name}
- CWE metadata: {cwe}

Code:
```c
{func}
```

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


PROMPT_CODE_METADATA_NO_CWE = """A C/C++ function is being evaluated for security vulnerability detection. CWE/CVE metadata is intentionally hidden for this bias-control experiment.

Metadata:
- project: {project}
- file name: {file_name}

Code:
```c
{func}
```

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


PROMPT_CWE_ONLY = """A C/C++ function is being evaluated for security vulnerability detection. The function code is intentionally hidden for this bias-control experiment.

CWE metadata:
- {cwe}

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


PROMPT_SKELETON = """Please analyze the following code skeleton. The function body has been removed for this bias-control experiment.

Function skeleton and coarse statistics:
```c
{skeleton}
```
- line count: {line_count}
- character length: {char_len}

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


VARIANTS = {
    "no_code_prior",
    "metadata_only",
    "metadata_no_cwe",
    "cwe_only",
    "skeleton_only",
    "code_zero_shot",
    "code_metadata",
    "code_metadata_no_cwe",
    "code_author_fewshot",
    "code_author_flipped_fewshot",
    "code_yes_only_fewshot",
    "code_no_only_fewshot",
    "no_code_author_fewshot",
    "no_code_flipped_fewshot",
}


def safe_int(value: Any, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


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


def first_signature_line(func: str) -> str:
    lines = [line.rstrip() for line in safe_str(func).splitlines() if line.strip()]
    if not lines:
        return "/* empty function */"
    for idx, line in enumerate(lines[:8]):
        if "{" in line:
            return "\n".join(lines[: idx + 1]) + "\n  /* body omitted */\n}"
    return lines[0] + "\n{\n  /* body omitted */\n}"


def flipped_label(label: str) -> str:
    return "NO" if label.strip().upper() == "YES" else "YES"


def fewshot_messages(variant: str) -> list[dict[str, str]]:
    yes_example = [
        {"role": "user", "content": ONESHOT_USER},
        {"role": "assistant", "content": ONESHOT_ASSISTANT},
    ]
    no_example = [
        {"role": "user", "content": TWOSHOT_USER},
        {"role": "assistant", "content": TWOSHOT_ASSISTANT},
    ]
    if variant in {"code_author_fewshot", "no_code_author_fewshot"}:
        return no_example + yes_example
    if variant in {"code_author_flipped_fewshot", "no_code_flipped_fewshot"}:
        return [
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": flipped_label(TWOSHOT_ASSISTANT)},
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": flipped_label(ONESHOT_ASSISTANT)},
        ]
    if variant == "code_yes_only_fewshot":
        return yes_example
    if variant == "code_no_only_fewshot":
        return no_example
    return []


def build_user_prompt(sample: dict, variant: str) -> str:
    func = safe_str(sample.get("func"))
    if variant in {"no_code_prior", "no_code_author_fewshot", "no_code_flipped_fewshot"}:
        return PROMPT_NO_CODE
    if variant == "metadata_only":
        return PROMPT_METADATA.format(
            project=sample.get("project", ""),
            file_name=sample.get("file_name", ""),
            cwe=sample.get("cwe", ""),
        )
    if variant == "metadata_no_cwe":
        return PROMPT_METADATA_NO_CWE.format(
            project=sample.get("project", ""),
            file_name=sample.get("file_name", ""),
        )
    if variant == "code_metadata":
        return PROMPT_CODE_METADATA.format(
            project=sample.get("project", ""),
            file_name=sample.get("file_name", ""),
            cwe=sample.get("cwe", ""),
            func=func,
        )
    if variant == "code_metadata_no_cwe":
        return PROMPT_CODE_METADATA_NO_CWE.format(
            project=sample.get("project", ""),
            file_name=sample.get("file_name", ""),
            func=func,
        )
    if variant == "cwe_only":
        return PROMPT_CWE_ONLY.format(cwe=sample.get("cwe", ""))
    if variant == "skeleton_only":
        return PROMPT_SKELETON.format(
            skeleton=first_signature_line(func),
            line_count=len(func.splitlines()),
            char_len=len(func),
        )
    return PROMPT_CODE.format(func=func)


def build_messages(sample: dict, variant: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYS_INST}]
    messages.extend(fewshot_messages(variant))
    messages.append({"role": "user", "content": build_user_prompt(sample, variant)})
    return messages


def compute_metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = unknown = 0
    for row in rows:
        target = safe_int(row.get("target"), -1)
        pred = row.get("prediction")
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
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "known": known,
        "unknown": unknown,
        "accuracy": (tp + tn) / known if known else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "directional_failure_index": recall - specificity,
        "yes_rate": (tp + fp) / known if known else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "idx",
        "target",
        "prediction",
        "correct",
        "variant",
        "project",
        "cwe",
        "file_name",
        "func_char_len",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "response",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bias-controlled prompt experiment.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outdir/empirical_study/bias_controlled")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--prompt_strategy", default="std_cls", choices=["std_cls", "cot"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_gen_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(Path(args.input), args.limit)
    output_jsonl = output_dir / f"{args.model}_{args.variant}.jsonl"
    output_csv = output_dir / f"{args.model}_{args.variant}_predictions.csv"
    output_summary = output_dir / f"{args.model}_{args.variant}_metrics.json"

    prompt_rows = []
    for sample in samples:
        messages = build_messages(sample, args.variant)
        prompt_rows.append(
            {
                "idx": sample.get("idx"),
                "target": sample.get("target"),
                "project": sample.get("project", ""),
                "cwe": sample.get("cwe", ""),
                "file_name": sample.get("file_name", ""),
                "func_char_len": len(safe_str(sample.get("func"))),
                "variant": args.variant,
                "messages": messages,
            }
        )

    if args.dry_run:
        for row in prompt_rows:
            row["response"] = "[DRY_RUN]"
            row["prediction"] = ""
            row["correct"] = ""
        write_jsonl(output_jsonl, prompt_rows)
        print(f"Wrote dry-run prompts: {output_jsonl}")
        return

    from model_api_clients import get_openai_chat, normalize_usage

    def worker(row: dict) -> dict:
        try:
            response, usage, messages, reasoning = get_openai_chat(
                {"messages": row["messages"]},
                args.model,
                args.prompt_strategy,
                args.temperature,
                args.max_gen_length,
                args.seed,
            )
        except Exception as exc:
            response, usage, messages, reasoning = f"[ERROR] {exc}", {}, row["messages"], None
        try:
            usage_row = normalize_usage(usage, args.model) if usage is not None else {}
        except Exception:
            usage_row = {}
        out = dict(row)
        out["messages"] = messages
        out["reasoning"] = reasoning
        out["usage"] = usage_row
        out["response"] = response
        pred = parse_prediction(response)
        target = safe_int(out.get("target"), -1)
        out["prediction"] = pred if pred in (0, 1) else ""
        out["correct"] = int(pred == target) if pred in (0, 1) and target in (0, 1) else ""
        out["prompt_tokens"] = usage_row.get("prompt_tokens", 0)
        out["completion_tokens"] = usage_row.get("completion_tokens", 0)
        out["total_tokens"] = usage_row.get("total_tokens", 0)
        return out

    output_rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.num_threads)) as executor:
        futures = [executor.submit(worker, row) for row in prompt_rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"bias:{args.variant}"):
            output_rows.append(future.result())
            write_jsonl(output_jsonl, output_rows)

    output_rows.sort(key=lambda row: safe_int(row.get("idx"), 10**18))
    metrics = compute_metrics(output_rows)
    metrics.update(
        {
            "input": args.input,
            "variant": args.variant,
            "model": args.model,
            "total_records": len(output_rows),
        }
    )
    write_jsonl(output_jsonl, output_rows)
    write_csv(output_csv, output_rows)
    output_summary.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
