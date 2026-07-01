#!/usr/bin/env python3
"""Retrieve patch-aware contrastive evidence for query functions.

This script consumes the paired indexes produced by build_patch_pair_index.py.
For each query chunk, it searches both historical sides:

  patch_pair_vuln.faiss  -> vulnerable/pre-fix changed regions
  patch_pair_fixed.faiss -> fixed/post-fix changed regions

The output keeps aligned patch pairs and reports whether the query is closer to
the vulnerable side or the fixed side of each historical repair.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from build_query_chunks_sim_rank import build_chunks, encode_chunks


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


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def code_identifiers(text: Any) -> set[str]:
    return {token for token in IDENT_RE.findall(safe_str(text)) if token not in STOP_IDENTIFIERS}


def code_calls(text: Any) -> set[str]:
    return {token for token in CALL_RE.findall(safe_str(text)) if token not in STOP_IDENTIFIERS}


def score_from_faiss(raw_score: float, use_ip: bool) -> float:
    # FAISS L2 returns smaller-is-better distances; convert to higher-is-better.
    return float(raw_score) if use_ip else -float(raw_score)


def load_patch_pair_metadata(path: Path) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            vector_id = item.get("vector_id")
            if vector_id is None:
                continue
            meta[int(vector_id)] = item
    return meta


def index_metric_is_ip(index) -> bool:
    import faiss

    return index.metric_type == faiss.METRIC_INNER_PRODUCT


def normalize_query_vectors(vectors, use_ip: bool):
    if not use_ip:
        return vectors
    import faiss

    faiss.normalize_L2(vectors)
    return vectors


def search_index(index, vectors, topk: int, use_ip: bool) -> list[dict[int, dict]]:
    distances, ids = index.search(vectors, topk)
    all_hits: list[dict[int, dict]] = []
    for row_scores, row_ids in zip(distances, ids):
        hits: dict[int, dict] = {}
        for rank, (raw_score, raw_id) in enumerate(zip(row_scores, row_ids), start=1):
            vector_id = int(raw_id)
            if vector_id < 0:
                continue
            hits[vector_id] = {
                "rank": rank,
                "raw_score": float(raw_score),
                "score": score_from_faiss(float(raw_score), use_ip),
            }
        all_hits.append(hits)
    return all_hits


def compact_pair_metadata(pair: dict) -> dict:
    return {
        "vector_id": pair.get("vector_id"),
        "pair_id": pair.get("pair_id"),
        "project": pair.get("project"),
        "commit_id": pair.get("commit_id"),
        "file_name": pair.get("file_name"),
        "cwe": pair.get("cwe"),
        "vuln_idx": pair.get("vuln_idx"),
        "fixed_idx": pair.get("fixed_idx"),
        "vuln_signature": pair.get("vuln_signature"),
        "fixed_signature": pair.get("fixed_signature"),
        "vuln_changed_code": pair.get("vuln_changed_code", ""),
        "fixed_changed_code": pair.get("fixed_changed_code", ""),
        "vuln_regions": pair.get("vuln_regions", []),
        "fixed_regions": pair.get("fixed_regions", []),
        "patch_diff": pair.get("patch_diff", ""),
        "repair_signature": pair.get("repair_signature", ""),
        "repair_signal": pair.get("repair_signal", ""),
        "repair_added_lines": pair.get("repair_added_lines", []),
        "repair_removed_lines": pair.get("repair_removed_lines", []),
        "repair_added_calls": pair.get("repair_added_calls", []),
        "repair_removed_calls": pair.get("repair_removed_calls", []),
        "repair_added_identifiers": pair.get("repair_added_identifiers", []),
        "repair_removed_identifiers": pair.get("repair_removed_identifiers", []),
        "has_pair_diff": pair.get("has_pair_diff", ""),
        "has_fixed_added_lines": pair.get("has_fixed_added_lines", ""),
        "repair_evidence_source": pair.get("repair_evidence_source", ""),
    }


def build_pair_matches(
    vuln_hits: dict[int, dict],
    fixed_hits: dict[int, dict],
    metadata: dict[int, dict],
    max_pairs: int,
    query_code: str = "",
) -> list[dict]:
    candidate_ids = sorted(set(vuln_hits) | set(fixed_hits))
    matches = []
    for vector_id in candidate_ids:
        pair = metadata.get(vector_id)
        if pair is None:
            continue
        vuln_hit = vuln_hits.get(vector_id)
        fixed_hit = fixed_hits.get(vector_id)
        vuln_score = safe_float(vuln_hit.get("score")) if vuln_hit else None
        fixed_score = safe_float(fixed_hit.get("score")) if fixed_hit else None

        if vuln_score is not None and fixed_score is not None:
            contrast_margin = vuln_score - fixed_score
        elif vuln_score is not None:
            contrast_margin = abs(vuln_score)
        elif fixed_score is not None:
            contrast_margin = -abs(fixed_score)
        else:
            continue

        pair_relevance = max(
            score for score in (vuln_score, fixed_score) if score is not None
        )
        match = compact_pair_metadata(pair)
        repair_text = "\n".join(
            [
                safe_str(match.get("repair_signature")),
                safe_str(match.get("patch_diff")),
                safe_str(match.get("fixed_changed_code")),
                safe_str(match.get("vuln_changed_code")),
            ]
        )
        query_tokens = code_identifiers(query_code)
        repair_tokens = code_identifiers(repair_text)
        query_calls = code_calls(query_code)
        repair_calls = code_calls(repair_text)
        match.update(
            {
                "vulnerable_side_score": vuln_score,
                "fixed_side_score": fixed_score,
                "contrast_margin": contrast_margin,
                "abs_contrast_margin": abs(contrast_margin),
                "pair_relevance": pair_relevance,
                "closer_side": "vulnerable" if contrast_margin > 0 else "fixed",
                "both_sides_retrieved": vuln_hit is not None and fixed_hit is not None,
                "vulnerable_side_rank": vuln_hit.get("rank") if vuln_hit else None,
                "fixed_side_rank": fixed_hit.get("rank") if fixed_hit else None,
                "repair_token_overlap": len(query_tokens & repair_tokens),
                "repair_call_overlap": len(query_calls & repair_calls),
                "has_real_repair": as_bool(match.get("has_pair_diff")) or bool(match.get("patch_diff")),
            }
        )
        matches.append(match)

    matches.sort(
        key=lambda item: (
            -int(bool(item.get("has_real_repair"))),
            -int(bool(item.get("has_fixed_added_lines"))),
            -int(item.get("repair_call_overlap", 0)),
            -int(item.get("repair_token_overlap", 0)),
            -float(item.get("pair_relevance", 0.0)),
            -float(item.get("abs_contrast_margin", 0.0)),
            int(item.get("vector_id", 0)),
        )
    )
    for rank, match in enumerate(matches[:max_pairs], start=1):
        match["patch_pair_rank"] = rank
    return matches[:max_pairs]


def chunk_rank_score(matches: list[dict], mode: str) -> float:
    if not matches:
        return 0.0
    if mode == "relevance":
        return max(float(match.get("pair_relevance", 0.0)) for match in matches)
    if mode == "vuln_margin":
        return max(float(match.get("contrast_margin", 0.0)) for match in matches)
    # contrast: prioritize chunks that have strong historical before/after separation.
    return max(float(match.get("abs_contrast_margin", 0.0)) for match in matches)


def selected_indices_by_score(scores: list[float], top_n: int) -> list[int]:
    ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    if top_n and top_n > 0:
        ranked = ranked[:top_n]
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Build patch-aware contrastive query retrieval results.")
    parser.add_argument("--input", required=True, help="PrimeVul labeled query JSONL")
    parser.add_argument("--patch-metadata", required=True, help="patch_pair_metadata.jsonl")
    parser.add_argument("--vuln-index", required=True, help="patch_pair_vuln.faiss")
    parser.add_argument("--fixed-index", required=True, help="patch_pair_fixed.faiss")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("SLICERAG_EMBED_MODEL", "microsoft/graphcodebert-base"),
        help="Embedding model name or path. Override with SLICERAG_EMBED_MODEL for local checkpoints.",
    )
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="")
    parser.add_argument("--faiss-topk", type=int, default=50, help="Top hits per side before pair union.")
    parser.add_argument("--max-patch-pairs", type=int, default=3, help="Patch pairs kept per selected query chunk.")
    parser.add_argument("--top-n", type=int, default=2, help="Query chunks kept per target function.")
    parser.add_argument(
        "--ranking-score",
        choices=["contrast", "relevance", "vuln_margin"],
        default="contrast",
        help="How to rank query chunks before keeping --top-n.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write query chunk metadata only; skip embedding/search.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_metadata_path = output_dir / "query_patch_chunk_metadata.jsonl"
    results_path = output_dir / "query_patch_contrast_results.jsonl"

    patch_metadata = load_patch_pair_metadata(Path(args.patch_metadata))

    if not args.dry_run:
        import faiss
        import numpy as np
        from transformers import AutoModel, AutoTokenizer

        vuln_index = faiss.read_index(str(args.vuln_index))
        fixed_index = faiss.read_index(str(args.fixed_index))
        if vuln_index.ntotal != fixed_index.ntotal:
            raise ValueError(f"Index size mismatch: vuln={vuln_index.ntotal}, fixed={fixed_index.ntotal}")
        if vuln_index.d != fixed_index.d:
            raise ValueError(f"Index dim mismatch: vuln={vuln_index.d}, fixed={fixed_index.d}")
        use_ip = index_metric_is_ip(vuln_index)
        if use_ip != index_metric_is_ip(fixed_index):
            raise ValueError("Vulnerable and fixed indexes must use the same FAISS metric.")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        model.eval()
        if args.device:
            model.to(args.device)
    else:
        np = None
        vuln_index = fixed_index = None
        use_ip = False
        tokenizer = model = None

    chunk_id = 0
    vector_id = 0
    results = []
    chunk_rows = []

    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            chunks, chunk_id, line_count = build_chunks(sample, args.chunk_size, args.stride, chunk_id)
            for chunk in chunks:
                chunk["vector_id"] = vector_id
                vector_id += 1
                chunk_rows.append(chunk)

            if args.dry_run:
                continue
            if not chunks:
                continue

            texts = [chunk["code_clean"] for chunk in chunks]
            vectors = np.asarray(
                encode_chunks(texts, tokenizer, model, args.device, args.batch_size, args.max_length),
                dtype="float32",
            )
            if vectors.size == 0:
                continue
            if vectors.shape[1] != vuln_index.d:
                raise ValueError(f"Embedding dim mismatch: query={vectors.shape[1]}, index={vuln_index.d}")
            vectors = normalize_query_vectors(vectors, use_ip)

            vuln_hits_by_chunk = search_index(vuln_index, vectors, args.faiss_topk, use_ip)
            fixed_hits_by_chunk = search_index(fixed_index, vectors, args.faiss_topk, use_ip)

            chunk_items = []
            rank_scores = []
            for chunk, vuln_hits, fixed_hits in zip(chunks, vuln_hits_by_chunk, fixed_hits_by_chunk):
                matches = build_pair_matches(
                    vuln_hits,
                    fixed_hits,
                    patch_metadata,
                    args.max_patch_pairs,
                    query_code=chunk.get("code_clean", ""),
                )
                score = chunk_rank_score(matches, args.ranking_score)
                rank_scores.append(score)
                chunk_items.append(
                    {
                        "chunk": chunk,
                        "patch_pair_matches": matches,
                        "query_patch_score": score,
                        "top_contrast_margin": matches[0].get("contrast_margin") if matches else None,
                        "top_pair_relevance": matches[0].get("pair_relevance") if matches else None,
                        "top_closer_side": matches[0].get("closer_side") if matches else None,
                    }
                )

            selected = selected_indices_by_score(rank_scores, args.top_n)
            query_chunks = []
            for query_rank, idx in enumerate(selected, start=1):
                item = chunk_items[idx]
                chunk = dict(item["chunk"])
                chunk["query_rank"] = query_rank
                chunk["query_patch_score"] = item["query_patch_score"]
                chunk["top_contrast_margin"] = item["top_contrast_margin"]
                chunk["top_pair_relevance"] = item["top_pair_relevance"]
                chunk["top_closer_side"] = item["top_closer_side"]
                chunk["patch_pair_matches"] = item["patch_pair_matches"]
                query_chunks.append(chunk)

            results.append(
                {
                    "query_idx": sample.get("idx"),
                    "query_func_id": sample.get("idx"),
                    "query_func": sample.get("func", ""),
                    "query_target": sample.get("target"),
                    "project": sample.get("project", ""),
                    "cwe": sample.get("cwe", ""),
                    "func_hash": sample.get("func_hash", ""),
                    "line_count": line_count,
                    "chunk_size": args.chunk_size,
                    "stride": args.stride,
                    "ranking_score": args.ranking_score,
                    "query_chunks": query_chunks,
                }
            )

    with chunk_metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunk_rows:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if not args.dry_run:
        with results_path.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_query_records": len(results) if not args.dry_run else None,
        "num_query_chunks": len(chunk_rows),
        "num_patch_pairs": len(patch_metadata),
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "faiss_topk": args.faiss_topk,
        "max_patch_pairs": args.max_patch_pairs,
        "top_n": args.top_n,
        "ranking_score": args.ranking_score,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
