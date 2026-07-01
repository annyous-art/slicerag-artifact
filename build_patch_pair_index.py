#!/usr/bin/env python3
"""Build a patch-aware contrastive index from paired PrimeVul data.

Each indexed item is a vulnerable/fixed function pair from the same project,
commit, and file. The two FAISS indexes are aligned by vector_id:

  pair_vuln.faiss  -> vulnerable changed region embedding
  pair_fixed.faiss -> fixed changed region embedding

Metadata stores both sides so query-time retrieval can compare whether a target
chunk is closer to the vulnerable side or the fixed side of a historical patch.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_chunk_index import encode_chunks, init_faiss_index


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
STOP_IDENTIFIERS = {
    "if",
    "else",
    "for",
    "while",
    "switch",
    "case",
    "return",
    "sizeof",
    "static",
    "const",
    "struct",
    "int",
    "char",
    "void",
    "long",
    "short",
    "unsigned",
    "signed",
    "bool",
    "true",
    "false",
    "NULL",
    "nullptr",
}


def safe_int(value: Any, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_cwe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def align_labels(labels: list[int], num_lines: int) -> list[int]:
    if not labels:
        return [0] * num_lines
    labels = [safe_int(label, 0) for label in labels]
    if len(labels) == num_lines:
        return labels
    if len(labels) > num_lines:
        return labels[:num_lines]
    return labels + [0] * (num_lines - len(labels))


def function_signature(func: str) -> str:
    for line in (func or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:240]
    return ""


def contiguous_regions(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    regions = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            regions.append((start, prev))
            start = prev = idx
    regions.append((start, prev))
    return regions


def changed_region_text(
    record: dict,
    context_lines: int,
    max_regions: int,
    max_chars: int,
    allow_fallback: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    func = record.get("func", "") or ""
    lines = func.splitlines()
    labels = align_labels(record.get("labels") or [], len(lines))
    changed = [idx for idx, label in enumerate(labels) if safe_int(label, 0) == 1]
    regions = contiguous_regions(changed)[:max_regions]
    blocks: list[str] = []
    meta: list[dict[str, Any]] = []

    for region_id, (start, end) in enumerate(regions, start=1):
        block_start = max(0, start - context_lines)
        block_end = min(len(lines) - 1, end + context_lines)
        meta.append(
            {
                "region_id": region_id,
                "changed_start_line": start + 1,
                "changed_end_line": end + 1,
                "context_start_line": block_start + 1,
                "context_end_line": block_end + 1,
            }
        )
        block_lines = [f"// Region {region_id}: changed lines {start + 1}-{end + 1} with context"]
        for line_no in range(block_start, block_end + 1):
            marker = ">>" if labels[line_no] == 1 else "  "
            block_lines.append(f"{marker} L{line_no + 1}: {lines[line_no]}")
        blocks.append("\n".join(block_lines))

    if not blocks and allow_fallback:
        # Fallback for rare records without labels: use the signature and first lines.
        fallback = [function_signature(func)] + [line for line in lines[: min(12, len(lines))]]
        blocks.append("// No changed lines found; fallback context\n" + "\n".join(fallback))

    text = "\n\n".join(blocks)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "\n/* ... truncated changed-region text ... */"
    return text, meta


def code_identifiers(text: str) -> set[str]:
    return {token for token in IDENT_RE.findall(text or "") if token not in STOP_IDENTIFIERS}


def code_calls(text: str) -> set[str]:
    return {token for token in CALL_RE.findall(text or "") if token not in STOP_IDENTIFIERS}


def classify_repair_signal(added_text: str, removed_text: str) -> str:
    text = f"{added_text}\n{removed_text}".lower()
    signals = []
    if re.search(r"\bif\s*\(|\bassert\s*\(|\bcheck\b|\bguard\b", text):
        signals.append("added_guard")
    if re.search(r"\b(len|length|size|count|offset|index|idx|bound|limit|capacity|max|min)\b|[<>]=?", text):
        signals.append("bounds_or_shape_check")
    if re.search(r"\bnull\b|\bnullptr\b|!\s*[a-zA-Z_]", text):
        signals.append("null_check")
    if re.search(r"\breturn\b|\bgoto\b|\bbreak\b|\bcontinue\b|\berror\b|\bfail\b", text):
        signals.append("error_handling")
    if re.search(r"\bfree\b|\bdelete\b|\bunref\b|\brelease\b|\brefcount\b|\bcleanup\b|\bdestroy\b", text):
        signals.append("state_or_lifetime_repair")
    removed_calls = code_calls(removed_text)
    added_calls = code_calls(added_text)
    if added_calls and removed_calls and added_calls != removed_calls:
        signals.append("api_replacement")
    return "|".join(dict.fromkeys(signals)) or "unknown"


def strip_numbered_line(line: str) -> str:
    return re.sub(r"^[+\- ]{0,2}\s*[VF]?\d+:\s*", "", line).strip()


def pair_diff_region_text(
    vuln_func: str,
    fixed_func: str,
    context_lines: int,
    max_regions: int,
    max_chars: int,
) -> tuple[str, str, str, list[dict[str, Any]], list[str], list[str]]:
    vuln_lines = (vuln_func or "").splitlines()
    fixed_lines = (fixed_func or "").splitlines()
    matcher = difflib.SequenceMatcher(a=vuln_lines, b=fixed_lines, autojunk=False)
    combined_blocks = []
    vuln_blocks = []
    fixed_blocks = []
    meta = []
    all_removed: list[str] = []
    all_added: list[str] = []

    region_id = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        region_id += 1
        if region_id > max_regions:
            break

        removed = vuln_lines[i1:i2]
        added = fixed_lines[j1:j2]
        all_removed.extend(line.strip() for line in removed if line.strip())
        all_added.extend(line.strip() for line in added if line.strip())

        v_context_start = max(0, i1 - context_lines)
        v_context_end = min(len(vuln_lines), i2 + context_lines)
        f_context_start = max(0, j1 - context_lines)
        f_context_end = min(len(fixed_lines), j2 + context_lines)
        meta.append(
            {
                "region_id": region_id,
                "tag": tag,
                "vuln_start_line": i1 + 1,
                "vuln_end_line": i2,
                "fixed_start_line": j1 + 1,
                "fixed_end_line": j2,
                "vuln_context_start_line": v_context_start + 1,
                "vuln_context_end_line": v_context_end,
                "fixed_context_start_line": f_context_start + 1,
                "fixed_context_end_line": f_context_end,
                "removed_line_count": len(removed),
                "added_line_count": len(added),
            }
        )

        v_lines = [f"// Diff Region {region_id}: vulnerable-side {tag} lines with context"]
        for line_no in range(v_context_start, v_context_end):
            marker = "--" if i1 <= line_no < i2 else "  "
            v_lines.append(f"{marker} V{line_no + 1}: {vuln_lines[line_no]}")
        vuln_block = "\n".join(v_lines)
        vuln_blocks.append(vuln_block)

        f_lines = [f"// Diff Region {region_id}: fixed-side repair lines with context"]
        if f_context_start == f_context_end and not added:
            f_lines.append("// No added fixed-side lines; repair may be deletion-only.")
        for line_no in range(f_context_start, f_context_end):
            marker = "++" if j1 <= line_no < j2 else "  "
            f_lines.append(f"{marker} F{line_no + 1}: {fixed_lines[line_no]}")
        fixed_block = "\n".join(f_lines)
        fixed_blocks.append(fixed_block)

        combined_blocks.append(
            "\n".join(
                [
                    f"// Patch Diff Region {region_id}: {tag}",
                    "Vulnerable-side removed/changed code:",
                    vuln_block,
                    "",
                    "Fixed-side added/repaired code:",
                    fixed_block,
                ]
            )
        )

    def join_and_truncate(blocks: list[str]) -> str:
        text = "\n\n".join(blocks)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "\n/* ... truncated patch diff text ... */"
        return text

    return (
        join_and_truncate(combined_blocks),
        join_and_truncate(vuln_blocks),
        join_and_truncate(fixed_blocks),
        meta,
        all_added,
        all_removed,
    )


def repair_signature_text(added_lines: list[str], removed_lines: list[str], max_lines: int = 40) -> str:
    added_text = "\n".join(added_lines)
    removed_text = "\n".join(removed_lines)
    signal = classify_repair_signal(added_text, removed_text)
    added_calls = sorted(code_calls(added_text))
    removed_calls = sorted(code_calls(removed_text))
    added_ids = sorted(code_identifiers(added_text))[:50]
    removed_ids = sorted(code_identifiers(removed_text))[:50]
    lines = [
        f"Repair signal: {signal}",
        f"Added calls: {', '.join(added_calls) if added_calls else 'NONE'}",
        f"Removed calls: {', '.join(removed_calls) if removed_calls else 'NONE'}",
        f"Added identifiers: {', '.join(added_ids) if added_ids else 'NONE'}",
        f"Removed identifiers: {', '.join(removed_ids) if removed_ids else 'NONE'}",
        "Added/repaired lines:",
    ]
    lines.extend(added_lines[:max_lines] if added_lines else ["NONE"])
    lines.append("Removed/vulnerable lines:")
    lines.extend(removed_lines[:max_lines] if removed_lines else ["NONE"])
    return "\n".join(lines)


def embedding_text(record: dict, changed_text: str, repair_signature: str = "") -> str:
    return "\n".join(
        [
            f"Project: {record.get('project', '')}",
            f"CWE: {normalize_cwe(record.get('cwe'))}",
            f"Function: {function_signature(record.get('func', '') or '')}",
            "Repair Signature:",
            repair_signature,
            "Changed Region:",
            changed_text,
        ]
    )


def group_key(record: dict) -> tuple:
    return (
        record.get("project", ""),
        record.get("commit_id", ""),
        record.get("file_name", ""),
    )


def choose_pairs(records: list[dict], pairing: str) -> list[tuple[dict, dict]]:
    vulns = [record for record in records if safe_int(record.get("target")) == 1]
    fixed = [record for record in records if safe_int(record.get("target")) == 0]
    if not vulns or not fixed:
        return []
    if pairing == "cartesian":
        return [(vuln, fix) for vuln in vulns for fix in fixed]

    pairs = []
    used_fixed: set[int] = set()
    for vuln in sorted(vulns, key=lambda item: str(item.get("idx", ""))):
        ranked = sorted(
            enumerate(fixed),
            key=lambda item: (
                item[0] in used_fixed,
                abs(len(vuln.get("func", "") or "") - len(item[1].get("func", "") or "")),
                str(item[1].get("idx", "")),
            ),
        )
        fix_idx, fix = ranked[0]
        used_fixed.add(fix_idx)
        pairs.append((vuln, fix))
    return pairs


def load_adjacent_pairs(input_path: Path) -> list[tuple[dict, dict]]:
    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) % 2 != 0:
        raise ValueError(f"Adjacent pairing expects an even number of rows, got {len(rows)}")

    pairs = []
    bad_examples = []
    for offset in range(0, len(rows), 2):
        vuln = rows[offset]
        fixed = rows[offset + 1]
        if safe_int(vuln.get("target")) == 1 and safe_int(fixed.get("target")) == 0:
            pairs.append((vuln, fixed))
        else:
            bad_examples.append(
                {
                    "offset": offset,
                    "first_idx": vuln.get("idx"),
                    "second_idx": fixed.get("idx"),
                    "first_target": vuln.get("target"),
                    "second_target": fixed.get("target"),
                }
            )
    if bad_examples:
        raise ValueError(
            "Adjacent pairing expects every two rows to be target=1 then target=0. "
            f"Bad examples: {bad_examples[:5]}"
        )
    return pairs


def load_grouped_pairs(input_path: Path, pairing: str) -> list[tuple[dict, dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            groups[group_key(record)].append(record)
    pairs = []
    for group_records in groups.values():
        pairs.extend(choose_pairs(group_records, pairing))
    return pairs


def load_pairs(input_path: Path, pairing: str) -> list[tuple[dict, dict]]:
    if pairing == "adjacent":
        return load_adjacent_pairs(input_path)
    return load_grouped_pairs(input_path, pairing)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_vectors_to_index(texts: list[str], tokenizer, model, args, index):
    import numpy as np
    import faiss

    vectors = encode_chunks(texts, tokenizer, model, args.device, args.batch_size, args.max_length, model_type=args.model_type)
    mat = np.asarray(vectors, dtype="float32")
    if mat.size == 0:
        return index
    if index is None:
        index = init_faiss_index(mat.shape[1], args.use_ip)
    if args.use_ip:
        faiss.normalize_L2(mat)
    index.add(mat)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patch-aware paired vuln/fix contrastive index.")
    parser.add_argument("--input", required=True, help="PrimeVul paired labeled train JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("SLICERAG_EMBED_MODEL", "microsoft/graphcodebert-base"),
        help="Embedding model name or path. Override with SLICERAG_EMBED_MODEL for local checkpoints.",
    )
    parser.add_argument("--model-type", choices=["auto", "codet5", "codebert", "graphcodebert"], default="graphcodebert")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--use-ip", action="store_true")
    parser.add_argument(
        "--pairing",
        choices=["adjacent", "closest", "cartesian"],
        default="adjacent",
        help=(
            "How to recover vulnerable/fixed pairs. Use adjacent for PrimeVul paired JSONL "
            "where every two rows are target=1 then target=0. closest/cartesian are legacy "
            "metadata-grouped fallbacks by project+commit+file."
        ),
    )
    parser.add_argument("--context-lines", type=int, default=3)
    parser.add_argument("--max-regions", type=int, default=4)
    parser.add_argument("--max-region-chars", type=int, default=6000)
    parser.add_argument(
        "--repair-evidence-source",
        choices=["diff", "labels", "diff_fallback"],
        default="diff",
        help=(
            "Source for indexed repair evidence. diff extracts real vulnerable->fixed "
            "added/removed lines from the pair; labels uses line labels; diff_fallback "
            "uses diff when available and labels otherwise."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "patch_pair_metadata.jsonl"
    vuln_index_path = output_dir / "patch_pair_vuln.faiss"
    fixed_index_path = output_dir / "patch_pair_fixed.faiss"

    pairs = load_pairs(Path(args.input), args.pairing)
    metadata: list[dict] = []
    vuln_texts: list[str] = []
    fixed_texts: list[str] = []

    for vector_id, (vuln, fixed) in enumerate(pairs):
        vuln_label_changed, vuln_label_regions = changed_region_text(
            vuln, args.context_lines, args.max_regions, args.max_region_chars, allow_fallback=False
        )
        fixed_label_changed, fixed_label_regions = changed_region_text(
            fixed, args.context_lines, args.max_regions, args.max_region_chars, allow_fallback=False
        )
        patch_diff, diff_vuln_changed, diff_fixed_changed, diff_regions, added_lines, removed_lines = pair_diff_region_text(
            vuln.get("func", "") or "",
            fixed.get("func", "") or "",
            args.context_lines,
            args.max_regions,
            args.max_region_chars,
        )
        repair_signature = repair_signature_text(added_lines, removed_lines)
        has_pair_diff = bool(diff_regions)
        has_fixed_added_lines = bool(added_lines)

        if args.repair_evidence_source == "labels":
            vuln_changed = vuln_label_changed or diff_vuln_changed
            fixed_changed = fixed_label_changed or diff_fixed_changed
            vuln_regions = vuln_label_regions
            fixed_regions = fixed_label_regions
            evidence_source = "labels"
        elif args.repair_evidence_source == "diff_fallback":
            vuln_changed = diff_vuln_changed or vuln_label_changed
            fixed_changed = diff_fixed_changed or fixed_label_changed
            vuln_regions = diff_regions or vuln_label_regions
            fixed_regions = diff_regions or fixed_label_regions
            evidence_source = "diff" if diff_regions else "labels"
        else:
            vuln_changed = diff_vuln_changed
            fixed_changed = diff_fixed_changed
            vuln_regions = diff_regions
            fixed_regions = diff_regions
            evidence_source = "diff"

        if not vuln_changed:
            vuln_changed, vuln_regions = changed_region_text(
                vuln, args.context_lines, args.max_regions, args.max_region_chars, allow_fallback=True
            )
        if not fixed_changed:
            fixed_changed, fixed_regions = changed_region_text(
                fixed, args.context_lines, args.max_regions, args.max_region_chars, allow_fallback=True
            )

        vuln_embed = embedding_text(vuln, vuln_changed, repair_signature)
        fixed_embed = embedding_text(fixed, fixed_changed, repair_signature)
        vuln_texts.append(vuln_embed)
        fixed_texts.append(fixed_embed)
        metadata.append(
            {
                "vector_id": vector_id,
                "pair_id": f"{vuln.get('idx')}_{fixed.get('idx')}",
                "project": vuln.get("project", fixed.get("project", "")),
                "commit_id": vuln.get("commit_id", fixed.get("commit_id", "")),
                "file_name": vuln.get("file_name", fixed.get("file_name", "")),
                "cwe": normalize_cwe(vuln.get("cwe") or fixed.get("cwe")),
                "vuln_idx": vuln.get("idx"),
                "fixed_idx": fixed.get("idx"),
                "vuln_func_hash": vuln.get("func_hash"),
                "fixed_func_hash": fixed.get("func_hash"),
                "vuln_signature": function_signature(vuln.get("func", "") or ""),
                "fixed_signature": function_signature(fixed.get("func", "") or ""),
                "vuln_changed_code": vuln_changed,
                "fixed_changed_code": fixed_changed,
                "vuln_regions": vuln_regions,
                "fixed_regions": fixed_regions,
                "patch_diff": patch_diff,
                "repair_signature": repair_signature,
                "repair_signal": classify_repair_signal("\n".join(added_lines), "\n".join(removed_lines)),
                "repair_added_lines": added_lines,
                "repair_removed_lines": removed_lines,
                "repair_added_calls": sorted(code_calls("\n".join(added_lines))),
                "repair_removed_calls": sorted(code_calls("\n".join(removed_lines))),
                "repair_added_identifiers": sorted(code_identifiers("\n".join(added_lines)))[:100],
                "repair_removed_identifiers": sorted(code_identifiers("\n".join(removed_lines)))[:100],
                "has_pair_diff": has_pair_diff,
                "has_fixed_added_lines": has_fixed_added_lines,
                "repair_evidence_source": evidence_source,
                "vuln_embedding_text": vuln_embed,
                "fixed_embedding_text": fixed_embed,
                "vuln_func": vuln.get("func", ""),
                "fixed_func": fixed.get("func", ""),
            }
        )

    write_jsonl(metadata_path, metadata)
    summary = {
        "num_pairs": len(metadata),
        "pairing": args.pairing,
        "repair_evidence_source": args.repair_evidence_source,
        "pairs_with_diff": sum(1 for row in metadata if row.get("has_pair_diff")),
        "pairs_with_fixed_added_lines": sum(1 for row in metadata if row.get("has_fixed_added_lines")),
        "repair_signal_counts": dict(Counter(row["repair_signal"] for row in metadata).most_common(30)),
        "project_counts": dict(Counter(row["project"] for row in metadata).most_common(30)),
        "cwe_counts": dict(Counter(row["cwe"] for row in metadata).most_common(30)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Wrote metadata: {metadata_path}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    from transformers import AutoModel, AutoTokenizer
    import faiss

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    if args.device:
        model.to(args.device)

    vuln_index = None
    fixed_index = None
    for start in range(0, len(metadata), args.batch_size):
        vuln_index = add_vectors_to_index(vuln_texts[start : start + args.batch_size], tokenizer, model, args, vuln_index)
        fixed_index = add_vectors_to_index(fixed_texts[start : start + args.batch_size], tokenizer, model, args, fixed_index)

    if vuln_index is None or fixed_index is None:
        raise RuntimeError("No vectors were produced for patch-pair indexes.")
    faiss.write_index(vuln_index, str(vuln_index_path))
    faiss.write_index(fixed_index, str(fixed_index_path))
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote vulnerable-side index: {vuln_index_path}")
    print(f"Wrote fixed-side index: {fixed_index_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
