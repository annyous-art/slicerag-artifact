#!/usr/bin/env python3
"""Analyze patch-contrast error modes using PrimeVul paired diffs.

Outputs:
- hard negative verifier candidates for target=0 all-wrong cases
- semantic categories for target=1 all-wrong cases
- patch_contrast-as-feature summaries
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHODS = [
    "zero_shot",
    "yes_only",
    "author_no_yes",
    "old_rag",
    "positive_weighted_rag",
    "patch_contrast",
]


CATEGORY_PATTERNS = {
    "auth_config": [
        r"\bauth\b",
        r"\bpermission\b",
        r"\bcredential\b",
        r"\bpasswd\b",
        r"\bpassword\b",
        r"\btoken\b",
        r"\bkey\b",
        r"\bcert\b",
        r"\bconf\b",
        r"\bconfig\b",
        r"\bhost\b",
        r"\bhostname\b",
        r"\bverify\b",
        r"\brestricted\b",
        r"\bsanitize\b",
    ],
    "shape_invariant": [
        r"\bshape\b",
        r"\bdim\b",
        r"\brank\b",
        r"\bsize\b",
        r"\bstride\b",
        r"\bindex\b",
        r"\bindices\b",
        r"\bmatrix\b",
        r"\btensor\b",
        r"\bcoordinate\b",
    ],
    "assert_dos": [
        r"\bCHECK\b",
        r"\bDCHECK\b",
        r"\bassert\b",
        r"\bASSERT\b",
        r"\bpanic\b",
        r"\babort\b",
        r"\bunreachable\b",
        r"\bfatal\b",
        r"\bBUG_ON\b",
    ],
    "state_flag": [
        r"\bflag\b",
        r"\bstate\b",
        r"\bvalid\b",
        r"\bmode\b",
        r"\bstatus\b",
        r"\bmark\b",
        r"\bset_\w+",
        r"\bclear_\w+",
    ],
    "lifetime_ownership": [
        r"\bfree\b",
        r"\brelease\b",
        r"\bdelete\b",
        r"\bdestroy\b",
        r"\bcopy\b",
        r"\bclone\b",
        r"\brefcount\b",
        r"\bowner\b",
        r"\bowned\b",
        r"\bborrow\b",
        r"\buse[-_]?after[-_]?free\b",
        r"\balloc\b",
        r"\bmalloc\b",
        r"\bSAFEALLOC\b",
        r"\bmutex_destroy\b",
        r"\bcleanup\b",
    ],
    "bounds_size": [
        r"\bbounds?\b",
        r"\blen\b",
        r"\blength\b",
        r"\bsize\b",
        r"\boffset\b",
        r"\bindex\b",
        r"\bcount\b",
        r"\blimit\b",
        r"\bmax\b",
        r"\bmin\b",
        r">=",
        r"<=",
        r">\s*\d+",
        r"<\s*\d+",
        r"\bnum\b",
        r"\bid\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\barray\b",
    ],
    "null_check": [
        r"\bNULL\b",
        r"\bnullptr\b",
        r"==\s*0\b",
        r"!=\s*0\b",
        r"!\s*\w+",
    ],
    "type_cast_alignment": [
        r"\bcast\b",
        r"\bstatic_cast\b",
        r"\breinterpret_cast\b",
        r"\buint\d+_t\b",
        r"\bint\d+_t\b",
        r"\bssize_t\b",
        r"\buintptr_t\b",
        r"\balign\b",
        r"\bendian\b",
        r"ReadProperty",
        r"\bwcstombs\b",
        r"\butf8\b",
    ],
    "error_handling": [
        r"\berror\b",
        r"\bInvalidArgument\b",
        r"\breturn\b",
        r"\bgoto\b",
        r"\bfail\b",
        r"\bbreak\b",
        r"\bcontinue\b",
        r"\bEINVAL\b",
        r"\bENOMEM\b",
    ],
    "parser_input": [
        r"\bread\b",
        r"\bparse\b",
        r"\bdecode\b",
        r"\bpacket\b",
        r"\bpayload\b",
        r"\bclient\b",
        r"\bstream\b",
        r"\bimage\b",
        r"\bfile\b",
        r"\btag\b",
        r"\bbitstream\b",
        r"\bbs\b",
        r"\bnalu\b",
        r"\bpps\b",
        r"\bsps\b",
    ],
    "crypto_compare": [
        r"\bcrypto\b",
        r"\bmemneq\b",
        r"\bmemcmp\b",
        r"\bstrncmp\b",
        r"\bhash\b",
        r"\bdigest\b",
    ],
    "randomness_entropy": [
        r"\brand\b",
        r"\brandom\b",
        r"\bentropy\b",
        r"\bjiffies\b",
        r"\birq\b",
        r"\bseed\b",
    ],
}


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "idx",
        "paired_idx",
        "target",
        "project",
        "cwe",
        "group",
        "primary_category",
        "categories",
        "repair_signal",
        "has_added_guard",
        "has_added_error_handling",
        "has_api_replacement",
        "has_state_or_lifetime_repair",
        "changed_line_count",
        "changed_region_count",
        "char_len",
        "line_count",
        "patch_margin",
        "patch_closer_side",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def adjacent_pair_map(records: list[dict]) -> dict[str, str]:
    pair = {}
    for offset in range(0, len(records), 2):
        if offset + 1 >= len(records):
            break
        first = records[offset]
        second = records[offset + 1]
        pair[str(first.get("idx"))] = str(second.get("idx"))
        pair[str(second.get("idx"))] = str(first.get("idx"))
    return pair


def label_stats(record: dict) -> tuple[int, int, float]:
    labels = [safe_int(item, 0) or 0 for item in (record.get("labels") or [])]
    changed = sum(1 for item in labels if item == 1)
    regions = 0
    prev = 0
    for item in labels:
        if item == 1 and prev != 1:
            regions += 1
        prev = item
    ratio = changed / max(1, len(labels))
    return changed, regions, ratio


def changed_lines(record: dict, max_lines: int = 20) -> list[str]:
    lines = (record.get("func") or "").splitlines()
    labels = [safe_int(item, 0) or 0 for item in (record.get("labels") or [])]
    out = []
    for line_no, (line, label) in enumerate(zip(lines, labels), start=1):
        if label == 1:
            out.append(f"L{line_no}: {line.strip()}")
            if len(out) >= max_lines:
                break
    return out


def paired_diff(vuln_record: dict, fixed_record: dict, context: int = 2, max_lines: int = 120) -> list[str]:
    diff = difflib.unified_diff(
        (vuln_record.get("func") or "").splitlines(),
        (fixed_record.get("func") or "").splitlines(),
        fromfile=f"vuln {vuln_record.get('idx')}",
        tofile=f"fixed {fixed_record.get('idx')}",
        n=context,
        lineterm="",
    )
    return list(diff)[:max_lines]


def diff_added_removed(diff_lines: list[str]) -> tuple[list[str], list[str]]:
    added = []
    removed = []
    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:].strip())
        elif line.startswith("-"):
            removed.append(line[1:].strip())
    return added, removed


def regex_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def categories_for_text(text: str) -> list[str]:
    categories = []
    for category, patterns in CATEGORY_PATTERNS.items():
        if regex_any(patterns, text):
            categories.append(category)
    return categories or ["other"]


def primary_category(categories: list[str]) -> str:
    priority = [
        "auth_config",
        "shape_invariant",
        "assert_dos",
        "lifetime_ownership",
        "state_flag",
        "bounds_size",
        "null_check",
        "type_cast_alignment",
        "crypto_compare",
        "randomness_entropy",
        "error_handling",
        "parser_input",
        "other",
    ]
    for category in priority:
        if category in categories:
            return category
    return categories[0] if categories else "other"


def repair_signal(added: list[str], removed: list[str]) -> dict:
    added_text = "\n".join(added)
    removed_text = "\n".join(removed)
    has_added_guard = regex_any(
        [
            r"\bif\s*\(",
            r"\bOP_REQUIRES\b",
            r"\bCHECK\b",
            r"\bassert\b",
            r"\bconfigASSERT\b",
            r"\breturn\b",
            r"\bbreak\b",
            r"\bcontinue\b",
            r">=",
            r"<=",
            r"\bNULL\b",
            r"\bnullptr\b",
        ],
        added_text,
    )
    has_added_error_handling = regex_any(CATEGORY_PATTERNS["error_handling"], added_text)
    has_api_replacement = bool(added and removed) and any(
        token in removed_text and token not in added_text
        for token in ["ml_get", "strncmp", "CHECK", "llvm_unreachable", "Acquire", "ReadProperty", "memcpy"]
    )
    if regex_any([r"\bcrypto_memneq\b", r"\bget_line_and_copy\b", r"\bcopy\b"], added_text):
        has_api_replacement = True
    has_state_or_lifetime_repair = regex_any(
        CATEGORY_PATTERNS["state_flag"] + CATEGORY_PATTERNS["lifetime_ownership"],
        added_text,
    )
    has_bounds_or_shape_repair = regex_any(
        CATEGORY_PATTERNS["bounds_size"] + CATEGORY_PATTERNS["shape_invariant"],
        added_text,
    )
    signal_names = []
    if has_added_guard:
        signal_names.append("added_guard")
    if has_added_error_handling:
        signal_names.append("added_error_handling")
    if has_api_replacement:
        signal_names.append("api_replacement")
    if has_state_or_lifetime_repair:
        signal_names.append("state_or_lifetime_repair")
    if has_bounds_or_shape_repair:
        signal_names.append("bounds_or_shape_repair")
    return {
        "repair_signal": "|".join(signal_names) if signal_names else "weak_or_unknown",
        "has_added_guard": int(has_added_guard),
        "has_added_error_handling": int(has_added_error_handling),
        "has_api_replacement": int(has_api_replacement),
        "has_state_or_lifetime_repair": int(has_state_or_lifetime_repair),
        "has_bounds_or_shape_repair": int(has_bounds_or_shape_repair),
    }


def correctness(row: dict, method: str) -> bool:
    return str(row.get(f"{method}_correct", "")) == "1"


def all_wrong(row: dict, methods: list[str]) -> bool:
    return all(not correctness(row, method) for method in methods)


def load_patch_quality(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row["idx"]): row for row in csv.DictReader(handle)}


def build_analysis_row(
    pred_row: dict,
    record: dict,
    paired_record: dict | None,
    patch_quality: dict[str, dict],
    group: str,
) -> dict:
    target = safe_int(pred_row.get("target"))
    vuln_record = record if target == 1 else paired_record
    fixed_record = record if target == 0 else paired_record
    diff_lines = paired_diff(vuln_record or {}, fixed_record or {}, context=2, max_lines=120)
    added, removed = diff_added_removed(diff_lines)
    text_for_category = "\n".join(added + removed + changed_lines(record, max_lines=20))
    categories = categories_for_text(text_for_category)
    changed_count, region_count, changed_ratio = label_stats(record)
    pq = patch_quality.get(str(pred_row.get("idx")), {})
    signal = repair_signal(added, removed)
    return {
        "idx": pred_row.get("idx"),
        "paired_idx": paired_record.get("idx") if paired_record else "",
        "target": target,
        "project": pred_row.get("project", record.get("project", "")),
        "cwe": pred_row.get("cwe", record.get("cwe", "")),
        "group": group,
        "primary_category": primary_category(categories),
        "categories": "|".join(categories),
        **signal,
        "changed_line_count": changed_count,
        "changed_region_count": region_count,
        "changed_line_ratio": round(changed_ratio, 6),
        "char_len": pred_row.get("char_len", len(record.get("func", "") or "")),
        "line_count": pred_row.get("line_count", len((record.get("func", "") or "").splitlines())),
        "patch_margin": pq.get("top_margin", ""),
        "patch_closer_side": pq.get("top_closer_side", ""),
        "patch_pair_project": pq.get("top_pair_project", ""),
        "patch_pair_cwe": pq.get("top_pair_cwe", ""),
        "pred_pattern": "|".join(str(pred_row.get(f"{method}_pred", "")) for method in METHODS),
        "changed_lines": "\n".join(changed_lines(record, max_lines=20)),
        "diff_excerpt": "\n".join(diff_lines),
        "added_lines": "\n".join(added[:40]),
        "removed_lines": "\n".join(removed[:40]),
    }


def verifier_prompt(row: dict, max_chars: int) -> str:
    diff_excerpt = row.get("diff_excerpt", "")
    target_code = row.get("target_func", "")
    if len(target_code) > max_chars:
        target_code = target_code[:max_chars] + "\n/* ... truncated ... */"
    return f"""A first-stage vulnerability detector predicted YES for this target function.
Your task is to verify whether the patch repair is already present in the target function.

Historical paired diff:
```diff
{diff_excerpt}
```

Target function:
```c
{target_code}
```

Answer NO if the repair/check/API replacement/state reset shown by the patch is present and adequately guards the risky behavior.
Answer YES only if the repaired target still lacks the relevant guard or still preserves the vulnerable-side behavior.

Return JSON only:
{{"verdict": "YES" or "NO", "repair_present": true or false, "reason": "brief"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze patch contrast error modes.")
    parser.add_argument("--data", default="data/primevul_test_paired_labeled.jsonl")
    parser.add_argument("--per-sample", default="outdir/ensemble_error_patch_contrast/per_sample_predictions.csv")
    parser.add_argument("--patch-quality", default="outdir/query_patch_contrast_test870_vuln_margin/patch_contrast_quality.csv")
    parser.add_argument("--output-dir", default="outdir/patch_error_modes")
    parser.add_argument("--max-verifier-code-chars", type=int, default=30000)
    args = parser.parse_args()

    records = load_jsonl(Path(args.data))
    by_idx = {str(record.get("idx")): record for record in records}
    pair_map = adjacent_pair_map(records)
    pred_rows = load_csv(Path(args.per_sample))
    patch_quality = load_patch_quality(Path(args.patch_quality)) if args.patch_quality else {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target0_hard_negative = []
    target1_semantic = []
    patch_feature_rows = []

    for pred_row in pred_rows:
        idx = str(pred_row.get("idx"))
        record = by_idx.get(idx)
        if record is None:
            continue
        paired_record = by_idx.get(pair_map.get(idx, ""))
        if all_wrong(pred_row, METHODS):
            target = safe_int(pred_row.get("target"))
            if target == 0:
                row = build_analysis_row(pred_row, record, paired_record, patch_quality, "target0_all_wrong_hard_negative")
                row["target_func"] = record.get("func", "")
                target0_hard_negative.append(row)
            elif target == 1:
                target1_semantic.append(
                    build_analysis_row(pred_row, record, paired_record, patch_quality, "target1_all_wrong_semantic")
                )

        pq = patch_quality.get(idx, {})
        patch_feature_rows.append(
            {
                "idx": idx,
                "target": pred_row.get("target"),
                "project": pred_row.get("project"),
                "cwe": pred_row.get("cwe"),
                "patch_contrast_pred": pred_row.get("patch_contrast_pred"),
                "patch_contrast_correct": pred_row.get("patch_contrast_correct"),
                "yes_only_pred": pred_row.get("yes_only_pred"),
                "yes_only_correct": pred_row.get("yes_only_correct"),
                "patch_margin": pq.get("top_margin", ""),
                "patch_closer_side": pq.get("top_closer_side", ""),
                "patch_pair_relevance": pq.get("top_pair_relevance", ""),
                "patch_pair_project": pq.get("top_pair_project", ""),
                "patch_pair_cwe": pq.get("top_pair_cwe", ""),
                "patch_feature_role": (
                    "rescues_yes_only_error"
                    if pred_row.get("patch_contrast_correct") == "1" and pred_row.get("yes_only_correct") != "1"
                    else "hurts_yes_only_correct"
                    if pred_row.get("patch_contrast_correct") != "1" and pred_row.get("yes_only_correct") == "1"
                    else "agrees"
                ),
            }
        )

    # Do not keep full target_func in CSV; write verifier JSONL separately.
    target0_csv_rows = [{k: v for k, v in row.items() if k != "target_func"} for row in target0_hard_negative]
    write_csv(output_dir / "target0_all_wrong_hard_negative.csv", target0_csv_rows)
    write_csv(output_dir / "target1_all_wrong_semantic_categories.csv", target1_semantic)
    write_csv(output_dir / "patch_contrast_feature_rows.csv", patch_feature_rows)

    with (output_dir / "target0_hard_negative_verifier.jsonl").open("w", encoding="utf-8") as handle:
        for row in target0_hard_negative:
            out = {
                "idx": row["idx"],
                "paired_idx": row["paired_idx"],
                "target": row["target"],
                "project": row["project"],
                "cwe": row["cwe"],
                "primary_category": row["primary_category"],
                "repair_signal": row["repair_signal"],
                "prompt": verifier_prompt(row, args.max_verifier_code_chars),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    def count_field(rows: list[dict], key: str) -> dict:
        return dict(Counter(row.get(key, "") for row in rows).most_common())

    summary = {
        "target0_all_wrong_hard_negative": {
            "n": len(target0_hard_negative),
            "primary_category_counts": count_field(target0_csv_rows, "primary_category"),
            "repair_signal_counts": count_field(target0_csv_rows, "repair_signal"),
            "cwe_counts": count_field(target0_csv_rows, "cwe"),
            "project_counts": count_field(target0_csv_rows, "project"),
        },
        "target1_all_wrong_semantic": {
            "n": len(target1_semantic),
            "primary_category_counts": count_field(target1_semantic, "primary_category"),
            "repair_signal_counts": count_field(target1_semantic, "repair_signal"),
            "cwe_counts": count_field(target1_semantic, "cwe"),
            "project_counts": count_field(target1_semantic, "project"),
        },
        "patch_feature_role_counts": count_field(patch_feature_rows, "patch_feature_role"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
