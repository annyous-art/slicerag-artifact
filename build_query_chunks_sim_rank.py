#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import sys

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))



def remove_inline_comment(line: str) -> str:
    in_string = False
    result_chars = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"' and (i == 0 or line[i - 1] != '\\'):
            in_string = not in_string
        if not in_string and i + 1 < len(line) and line[i:i + 2] == "//":
            break
        result_chars.append(ch)
        i += 1
    return "".join(result_chars)


def remove_block_comment_inline(line: str) -> str:
    result = ""
    i = 0
    while i < len(line):
        if i + 1 < len(line) and line[i:i + 2] == "/*":
            end = line.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        result += line[i]
        i += 1
    return result


def clean_code(lines: List[str]) -> Tuple[List[str], List[int], List[int]]:
    """
    Clean code by removing comment text while preserving line structure.

    Returns:
        cleaned_lines: lines after comment removal; blank lines are preserved
        is_comment: flags for comment lines
        is_blank: flags for blank lines
    """
    cleaned_lines: List[str] = []
    is_comment: List[int] = []
    is_blank: List[int] = []

    in_block_comment = False

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            is_blank.append(1)
            is_comment.append(1 if in_block_comment else 0)
            cleaned_lines.append(line)
            continue
        else:
            is_blank.append(0)

        comment_flag = 0

        if in_block_comment:
            comment_flag = 1
            if "*/" in line:
                in_block_comment = False
                line = line.split("*/", 1)[1]
                line = remove_block_comment_inline(line)
            else:
                line = ""

        if not in_block_comment and "/*" in line:
            comment_flag = 1
            if "*/" in line:
                line = remove_block_comment_inline(line)
            else:
                in_block_comment = True
                line = line.split("/*", 1)[0]

        stripped_line = line.strip()
        if stripped_line.startswith("//"):
            comment_flag = 1
            line = ""
        elif "//" in line:
            comment_flag = 1
            line = remove_inline_comment(line)

        line = line.replace("\t", " ")
        line = re.sub(r"[ ]+", " ", line)
        cleaned_lines.append(line)
        is_comment.append(comment_flag)

    return cleaned_lines, is_comment, is_blank


def get_valid_lines_info(lines: List[str], is_comment: List[int], is_blank: List[int]) -> Tuple[List[int], List[str]]:
    """
    Collect non-comment, non-blank code lines.

    Returns:
        valid_indices: original line indices for valid code lines
        valid_lines: cleaned contents of valid code lines
    """
    valid_indices = []
    valid_lines = []

    for i, (line, comment, blank) in enumerate(zip(lines, is_comment, is_blank)):
        if blank == 0 and comment == 0:
            valid_indices.append(i)
            # Re-clean this line to ensure comments are fully removed.
            valid_line = line
            if "//" in valid_line:
                valid_line = remove_inline_comment(valid_line)
            valid_line = valid_line.replace("\t", " ")
            valid_line = re.sub(r"[ ]+", " ", valid_line).strip()
            valid_lines.append(valid_line)

    return valid_indices, valid_lines


def align_labels(labels: List[int], num_lines: int) -> List[int]:
    if not labels:
        return [0] * num_lines
    if len(labels) == num_lines:
        return labels
    if len(labels) > num_lines:
        return labels[:num_lines]
    return labels + [0] * (num_lines - len(labels))


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_evidence_role_from_values(function_target, chunk_label: int) -> str:
    target = safe_int(function_target)
    if target == 1 and chunk_label == 1:
        return "vulnerable_changed"
    if target == 0 and chunk_label == 1:
        return "fixed_changed"
    if target == 0:
        return "safe_background"
    if target == 1:
        return "vulnerable_context"
    return "unknown"


def get_evidence_polarity_from_role(role: str):
    if role == "vulnerable_changed":
        return 1
    if role in ("fixed_changed", "safe_background"):
        return 0
    return None


def build_chunks(sample: Dict, chunk_size: int, stride: int, chunk_id_start: int) -> Tuple[List[Dict], int, int]:
    func = sample.get("func", "") or ""
    labels = sample.get("labels") or []
    idx = sample.get("idx")
    project = sample.get("project", "")
    cwe = sample.get("cwe", "")
    func_hash = sample.get("func_hash", "")
    commit_id = sample.get("commit_id", "")
    target = sample.get("target", "")
    has_is_fixed_version = "is_fixed_version" in sample
    is_fixed_version = sample.get("is_fixed_version")

    lines = func.splitlines()
    cleaned_lines, is_comment_flags, is_blank_flags = clean_code(lines)
    is_brace_flags = [1 if ln.strip() in ("{", "}", "};") else 0 for ln in lines]

    labels = align_labels(labels, len(lines))

    # Collect valid code lines.
    valid_indices, valid_lines = get_valid_lines_info(lines, is_comment_flags, is_blank_flags)

    chunks_meta: List[Dict] = []
    chunk_id = chunk_id_start
    local_chunk_id = 0

    # Build chunks over valid code lines.
    for valid_i in range(0, len(valid_lines), stride):
        # Determine the valid-line range for the current chunk.
        chunk_valid_end = min(valid_i + chunk_size, len(valid_lines))

        # Backfill short tail chunks to the requested chunk size.
        if chunk_valid_end - valid_i < chunk_size:
            needed = chunk_size - (chunk_valid_end - valid_i)
            valid_i = max(0, valid_i - needed)
            chunk_valid_end = min(valid_i + chunk_size, len(valid_lines))

        chunk_valid_indices = valid_indices[valid_i:chunk_valid_end]
        chunk_valid_lines = valid_lines[valid_i:chunk_valid_end]

        if len(chunk_valid_lines) == 0:
            continue

        # -----------------------------
        # Raw code range used for display.
        # -----------------------------
        start_raw = max(0, chunk_valid_indices[0] - 2)
        end_raw = min(len(lines) - 1, chunk_valid_indices[-1] + 2)
        raw_lines = lines[start_raw:end_raw + 1]

        raw_line_ids = list(
            range(start_raw + 1, end_raw + 2)
        )

        # -----------------------------
        # Valid code range used for labels and evaluation.
        # This aligns one-to-one with code_clean.
        # -----------------------------
        chunk_line_ids = [
            idx + 1
            for idx in chunk_valid_indices
        ]

        # Compute chunk_label from labels on valid lines.
        chunk_labels = [labels[i] for i in chunk_valid_indices]
        chunk_label = 1 if any(chunk_labels) else 0
        evidence_role_value = get_evidence_role_from_values(target, chunk_label)
        evidence_polarity_value = get_evidence_polarity_from_role(evidence_role_value)

        # Record comment and blank line ids.
        comment_ids = [raw_line_ids[i] for i, v in enumerate(is_comment_flags[start_raw:end_raw + 1]) if v == 1]
        blank_ids = [raw_line_ids[i] for i, v in enumerate(is_blank_flags[start_raw:end_raw + 1]) if v == 1]
        brace_ids = [raw_line_ids[i] for i, v in enumerate(is_brace_flags[start_raw:end_raw + 1]) if v == 1]

        chunks_meta.append({
            "chunk_id": chunk_id,
            "file_id": idx,
            "project": project,
            "cwe": cwe,
            "func_hash": func_hash,
            "commit_id": commit_id,
            "func": func,
            "target": target,
            "function_target": target,
            **({"is_fixed_version": is_fixed_version} if has_is_fixed_version else {}),
            "function_id": idx,
            "chunk_seq": local_chunk_id,
            "start_line": start_raw + 1,
            "end_line": end_raw + 1,
            "raw_line_ids": raw_line_ids,
            "code_raw": "\n".join(raw_lines),
            "code_clean": "\n".join(chunk_valid_lines),  # valid code lines only
            "chunk_line_ids": chunk_line_ids,
            "line_labels": chunk_labels,
            "chunk_label": chunk_label,
            "evidence_role": evidence_role_value,
            "evidence_polarity": evidence_polarity_value,
            "is_comment": comment_ids,
            "is_blank": blank_ids,
            "is_brace": brace_ids,
            "valid_count": len(chunk_valid_lines),
        })
        chunk_id += 1
        local_chunk_id += 1

    return chunks_meta, chunk_id, len(lines)


def encode_chunks(texts: List[str], tokenizer, model, device: str, batch_size: int, max_length: int) -> "List[List[float]]":
    import torch

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            if device:
                tokens = {k: v.to(device) for k, v in tokens.items()}

            # Use encoder output for seq2seq models like CodeT5; otherwise use the model forward pass
            if bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False)):
                encoder_outputs = model.encoder(**tokens)
                last_hidden = encoder_outputs.last_hidden_state
            else:
                outputs = model(**tokens)
                last_hidden = getattr(outputs, "last_hidden_state", None)
                if last_hidden is None and isinstance(outputs, tuple) and len(outputs) > 0:
                    last_hidden = outputs[0]
            attention_mask = tokens["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = (last_hidden * mask).sum(1)
            sum_mask = mask.sum(1).clamp(min=1e-6)
            emb = sum_embeddings / sum_mask
            embeddings.extend(emb.cpu().numpy().tolist())

    return embeddings


def load_index_metadata(path: Path) -> Dict[int, Dict]:
    meta = {}
    with path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vid = item.get("vector_id")
            if vid is None:
                continue
            meta[int(vid)] = {
                "file_id": item.get("file_id"),
                "function_id": item.get("function_id"),
                    # full source function text (from build_chunk_index.py chunks_meta 'func')
                "func": item.get("func"),
                "function_target": item.get("function_target"),
                "project": item.get("project"),
                "cwe": item.get("cwe"),
                "func_hash": item.get("func_hash"),
                "is_fixed_version": item.get("is_fixed_version"),
                "commit_id": item.get("commit_id"),
                "chunk_seq": item.get("chunk_seq"),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
                "chunk_label": item.get("chunk_label"),
                "evidence_role": item.get("evidence_role"),
                "evidence_polarity": item.get("evidence_polarity"),
                "code_raw": item.get("code_raw"),
                "code_clean": item.get("code_clean"),
                "score_tfidf_raw": item.get("score_tfidf_raw"),
                "score_keyword_raw": item.get("score_keyword_raw"),
                "score_label_raw": item.get("score_label_raw"),
                "score_combined": item.get("score_combined"),
                "score_combined_tfidf_keyword": item.get("score_combined_tfidf_keyword"),
                "score_combined_tfidf_label": item.get("score_combined_tfidf_label"),
                "score_combined_keyword_label": item.get("score_combined_keyword_label"),
                "line_labels": item.get("line_labels"),
                "chunk_line_ids": item.get("chunk_line_ids"),
                "raw_line_ids": item.get("raw_line_ids"),
            }
    return meta


def build_retrieval_pool(index_meta: Dict[int, Dict], mode: str) -> List[int]:
    if mode == "all":
        return sorted(index_meta.keys())
    return sorted(
        vector_id
        for vector_id, meta in index_meta.items()
        if evidence_role(meta) == "vulnerable_changed"
    )


def build_index_vector_id_list(index_meta: Dict[int, Dict]) -> List[int]:
    return sorted(index_meta.keys())


def get_match_chunk_label(match: Dict):
    value = match.get("chunk_label", match.get("index_chunk_label"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_match_function_target(match: Dict):
    value = match.get("function_target")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evidence_role(match: Dict) -> str:
    """Classify a retrieved chunk using function label plus diff/change label.

    chunk_label means "line changed by vulnerable/fixed diff", not necessarily
    "this chunk is vulnerable". For fixed-side records, changed chunks are
    repaired/safe evidence.
    """
    role = match.get("evidence_role")
    if role:
        return str(role)

    chunk_label = get_match_chunk_label(match)
    function_target = get_match_function_target(match)
    if function_target == 1 and chunk_label == 1:
        return "vulnerable_changed"
    if function_target == 0 and chunk_label == 1:
        return "fixed_changed"
    if function_target == 0:
        return "safe_background"
    if function_target == 1:
        return "vulnerable_context"
    return "unknown"


def evidence_polarity(match: Dict):
    polarity = match.get("evidence_polarity")
    if polarity is not None:
        try:
            return int(polarity)
        except (TypeError, ValueError):
            pass

    role = evidence_role(match)
    if role == "vulnerable_changed":
        return 1
    if role in ("fixed_changed", "safe_background"):
        return 0
    return None


def keep_comparative_matches(matches: List[Dict], topk: int) -> List[Dict]:
    """Keep enough neighbors for positive-vs-negative comparison without storing all FAISS hits."""
    if topk <= 0:
        return matches

    matches = sorted(matches, key=lambda item: item.get("index_rank", 999))
    overall = matches[:topk]
    positive = [match for match in matches if evidence_polarity(match) == 1][:topk]
    negative = [match for match in matches if evidence_polarity(match) == 0][:topk]

    kept: Dict[tuple, Dict] = {}
    for match in overall + positive + negative:
        key = (match.get("index_faiss_id"), match.get("vector_id"))
        kept[key] = match
    return sorted(kept.values(), key=lambda item: item.get("index_rank", 999))


def select_top_n(scores: List[float], top_n: int) -> List[int]:
    if top_n <= 0 or not scores:
        return []
    ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    return ranked[: min(top_n, len(ranked))]


def choose_ranking_scores(
    chunk_scores_cosine: List[float],
    chunk_scores_weighted: List[float],
    chunk_scores_margin: List[float],
    chunk_scores_max: List[float],
    ranking_score: str,
) -> List[float]:
    if ranking_score == "margin":
        return chunk_scores_margin
    if ranking_score == "max":
        return chunk_scores_max
    if ranking_score == "cosine":
        return chunk_scores_cosine
    return chunk_scores_weighted


def group_selected_query_items(selected_query_items: List[Dict], query_meta_by_vector_id: Dict[int, Dict]) -> List[Dict]:
    grouped_items: Dict[int, Dict] = {}
    ordered_keys: List[int] = []

    for item in selected_query_items:
        query_meta = query_meta_by_vector_id.get(int(item["query_vector_id"]), {})
        query_func_id = int(query_meta.get("function_id") or query_meta.get("file_id") or item["query_vector_id"])
        query_idx = int(query_meta.get("file_id") or query_meta.get("idx") or item["query_vector_id"])
        group_key = query_func_id

        if group_key not in grouped_items:
            grouped_items[group_key] = {
                "query_idx": query_idx,
                "query_func_id": query_func_id,
                "query_func": query_meta.get("func"),
                "query_target": query_meta.get("target"),
                "query_project": query_meta.get("project", ""),
                "query_commit_id": query_meta.get("commit_id", ""),
                "query_chunks": [],
            }
            ordered_keys.append(group_key)

        query_chunk = dict(query_meta)
        index_matches: List[Dict] = []
        for match in item.get("index_matches", []):
            index_chunk = dict(match)
            if "vector_id" in index_chunk and "index_vector_id" not in index_chunk:
                index_chunk["index_vector_id"] = index_chunk.pop("vector_id")
            index_matches.append(
                {
                    "index_code_raw": index_chunk.get("code_raw"),
                        # full function/source from index metadata (if available)
                    "index_func": index_chunk.get("func"),
                    "index_code_clean": index_chunk.get("code_clean"),
                    "index_score_combined": index_chunk.get("score_combined"),
                    "index_score": index_chunk.get("score"),
                    "faiss_score": index_chunk.get("faiss_score"),
                    "index_rank": index_chunk.get("index_rank"),
                    "index_chunk": index_chunk,
                    "chunk_label": index_chunk.get("chunk_label"),
                    "index_chunk_label": index_chunk.get("index_chunk_label", index_chunk.get("chunk_label")),
                    "evidence_role": evidence_role(index_chunk),
                    "evidence_polarity": evidence_polarity(index_chunk),
                    "index_line_labels": index_chunk.get("index_line_labels", index_chunk.get("line_labels")),
                    "chunk_line_ids": index_chunk.get("chunk_line_ids"),
                    "project": index_chunk.get("project"),
                    "cwe": index_chunk.get("cwe"),
                    "func_hash": index_chunk.get("func_hash"),
                    "is_fixed_version": index_chunk.get("is_fixed_version"),
                    "function_target": index_chunk.get("function_target"),
                }
            )

        grouped_items[group_key]["query_chunks"].append(
            {
                "query_code_raw": query_chunk.get("code_raw"),
                "query_code_clean": query_chunk.get("code_clean"),
                "query_score_cosine": item["query_score_cosine"],
                "query_score_weighted": item["query_score_weighted"],
                "top_pos_score": item.get("top_pos_score", 0.0),
                "top_neg_score": item.get("top_neg_score", 0.0),
                "pos_neg_margin": item.get("pos_neg_margin", 0.0),
                "positive_count": item.get("positive_count", 0),
                "negative_count": item.get("negative_count", 0),
                "query_score": item["query_score"],
                "query_rank": item["query_rank"],
                "query_vector_id": item["query_vector_id"],
                "query_chunk_id": item["query_chunk_id"],
                "query_chunk": query_chunk,
                "index_matches": index_matches,
            }
        )

    grouped_results: List[Dict] = []
    for group_key in ordered_keys:
        grouped_item = grouped_items[group_key]
        grouped_item["query_chunks"] = sorted(
            grouped_item["query_chunks"],
            key=lambda chunk: chunk.get("query_rank", 0),
        )
        grouped_results.append(grouped_item)

    return grouped_results


def normalize_rows(vectors):
    import numpy as np

    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def encode_and_search(texts: List[str], tokenizer, model, device: str, batch_size: int, max_length: int, index, topk: int, use_ip: bool):
    import numpy as np

    vectors = np.asarray(encode_chunks(texts, tokenizer, model, device, batch_size, max_length), dtype="float32")
    if vectors.size == 0:
        return vectors, None, None
    if index is None:
        raise RuntimeError("FAISS index not loaded")
    if index.d != vectors.shape[1]:
        raise ValueError(f"Embedding dim mismatch: index {index.d} vs batch {vectors.shape[1]}")
    if use_ip:
        import faiss
        faiss.normalize_L2(vectors)
    distances, ids = index.search(vectors, topk)
    return vectors, distances, ids



def main() -> None:
    parser = argparse.ArgumentParser(description="Build query chunks and search FAISS index")
    parser.add_argument("--input", required=True, help="Path to labeled JSONL file for queries")
    parser.add_argument("--index", required=True, help="Path to FAISS index from training chunks")
    parser.add_argument("--output-dir", required=True, help="Output directory for query metadata and results")
    parser.add_argument(
        "--model",
        default=os.getenv("SLICERAG_EMBED_MODEL", "microsoft/graphcodebert-base"),
        help="Embedding model name or path. Override with SLICERAG_EMBED_MODEL for local checkpoints.",
    )
    parser.add_argument("--chunk-size", type=int, default=6, help="Chunk size")
    parser.add_argument("--stride", type=int, default=2, help="Chunk stride")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max input tokens kept for embedding; this truncates the encoder input, not generation output")
    parser.add_argument("--device", default="", help="Device for model, e.g. cpu or cuda")
    parser.add_argument("--topk", type=int, default=5, help="Top-k results per query")
    parser.add_argument(
        "--faiss-topk",
        type=int,
        default=50,
        help="How many nearest neighbors to retrieve before filtering/reranking. "
        "Keep this larger than --topk so positive/negative evidence is available.",
    )
    parser.add_argument("--top-n", type=int, default=2, help="How many query chunks to keep per sample after weighted scoring")
    parser.add_argument(
        "--ranking-score",
        choices=["margin", "max", "weighted", "cosine"],
        default="margin",
        help="Score used to rank query chunks before keeping top-n. margin uses top positive minus top negative; max uses best evidence score. weighted/cosine are kept for compatibility and use per-chunk max, not sums.",
    )
    parser.add_argument(
        "--index-score-field",
        choices=[
            "score_combined",
            "score_combined_tfidf_keyword",
            "score_combined_tfidf_label",
            "score_combined_keyword_label",
        ],
        default="score_combined",
        help="Index-side score field used by weighted reranking. Keep score_combined for the full model or switch to a pairwise ablation field.",
    )
    parser.add_argument("--index-metadata", help="Path to chunk_metadata.jsonl for lookup")
    parser.add_argument(
        "--retrieval-mode",
        choices=["positive", "all"],
        default="all",
        help="Restrict query matches to positive training chunks or keep all index chunks",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write metadata, skip model and search")
    parser.add_argument(
        "--eval-label-path",
        default="",
        help="Path to labeled JSONL (e.g. primevul_test_paired_labeled.jsonl) for evaluating chunk hit rate. "
        "If provided, prints top-1/top-K chunk vulnerability-line hit statistics after building.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "query_chunk_metadata.jsonl"
    results_path = output_dir / "query_results.jsonl"

    chunk_id = 0
    vector_id = 0
    total_lines = 0
    total_chunks = 0
    missing_idx = 0

    tokenizer = None
    model = None

    if not args.dry_run:
        import faiss
        index = faiss.read_index(str(args.index))
        use_ip = index.metric_type == faiss.METRIC_INNER_PRODUCT
        if not use_ip:
            print("WARNING: index is not inner-product based; cosine recall assumes the index was built with --use-ip")
    else:
        index = None
        use_ip = False

    index_meta = None
    index_vector_ids: List[int] = []
    if args.index_metadata:
        index_meta = load_index_metadata(Path(args.index_metadata))
        index_vector_ids = build_index_vector_id_list(index_meta)
    use_index_id_map = index_meta is not None and len(index_vector_ids) > 0

    retrieval_vector_set = None
    if args.retrieval_mode != "all":
        if index_meta is None:
            raise ValueError("--retrieval-mode requires --index-metadata")
        retrieval_vector_set = set(build_retrieval_pool(index_meta, args.retrieval_mode))

    search_topk = max(args.topk, args.faiss_topk)

    try:
        from tqdm import tqdm
        progress_iter = lambda it: tqdm(it, desc="Processing query samples", unit="sample")
    except Exception:
        progress_iter = lambda it: it

    selected_records: List[Dict] = []
    selected_query_items: List[Dict] = []

    with input_path.open("r", encoding="utf-8") as f_in:
        for line in progress_iter(f_in):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "idx" not in sample:
                missing_idx += 1
                sample["idx"] = None
            new_chunks, chunk_id, num_lines = build_chunks(sample, args.chunk_size, args.stride, chunk_id)
            total_lines += num_lines
            if not new_chunks:
                continue
            if args.dry_run:
                for entry in new_chunks:
                    entry["vector_id"] = vector_id
                    vector_id += 1
                    total_chunks += 1
                    selected_records.append(entry)
                continue

            if tokenizer is None or model is None:
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(args.model)
                model = AutoModel.from_pretrained(args.model)
                model.eval()
                if args.device:
                    model.to(args.device)

            vectors, distances, ids = encode_and_search(
                [entry["code_clean"] for entry in new_chunks],
                tokenizer,
                model,
                args.device,
                args.batch_size,
                args.max_length,
                index,
                search_topk,
                use_ip,
            )

            chunk_scores_cosine: List[float] = []
            chunk_scores_weighted: List[float] = []
            chunk_scores_margin: List[float] = []
            chunk_scores_max: List[float] = []
            chunks_with_matches: List[Dict] = []

            # Step 1: Build FAISS matches per query chunk
            faiss_matches_per_chunk: List[List[Dict]] = []
            for i, entry in enumerate(new_chunks):
                entry["vector_id"] = vector_id
                vector_id += 1
                total_chunks += 1

                matches = []
                row_ids = ids[i]
                row_scores = distances[i]
                for j in range(len(row_ids)):
                    faiss_id = int(row_ids[j])
                    if faiss_id < 0:
                        continue
                    vid = faiss_id
                    if use_index_id_map:
                        if faiss_id >= len(index_vector_ids):
                            continue
                        vid = index_vector_ids[faiss_id]
                    if retrieval_vector_set is not None and vid not in retrieval_vector_set:
                        continue
                    match = {
                        "index_faiss_id": faiss_id,
                        "vector_id": vid,
                        "faiss_score": float(row_scores[j]),
                        "score": float(row_scores[j]),  # backward compat
                    }
                    if index_meta is not None and vid in index_meta:
                        match.update(index_meta[vid])
                        match["index_chunk_label"] = index_meta[vid].get("chunk_label", 0)
                        match["index_line_labels"] = index_meta[vid].get("line_labels", [])
                    match["index_rank"] = len(matches) + 1
                    matches.append(match)
                faiss_matches_per_chunk.append(keep_comparative_matches(matches, args.topk))

            # Step 2: Compute positive/negative evidence metrics for each query chunk.
            # Query chunk ranking uses margin or max evidence instead of summing all top-k scores.
            for i, entry in enumerate(new_chunks):
                matches = faiss_matches_per_chunk[i]
                matches.sort(key=lambda x: x.get("index_rank", 999))

                pos_scores = [float(m.get("faiss_score", 0.0)) for m in matches if evidence_polarity(m) == 1]
                neg_scores = [float(m.get("faiss_score", 0.0)) for m in matches if evidence_polarity(m) == 0]

                top_pos_score = max(pos_scores) if pos_scores else 0.0
                top_neg_score = max(neg_scores) if neg_scores else 0.0
                pos_neg_margin = top_pos_score - top_neg_score
                max_score = max((float(m.get("faiss_score", 0.0)) for m in matches), default=0.0)
                weighted_score = max(
                    (
                        float(m.get("faiss_score", 0.0)) * float(m.get(args.index_score_field, 1.0) or 1.0)
                        for m in matches
                    ),
                    default=0.0,
                )

                chunk_scores_cosine.append(max_score)
                chunk_scores_weighted.append(weighted_score)
                chunk_scores_margin.append(pos_neg_margin)
                chunk_scores_max.append(max_score)
                chunks_with_matches.append(
                    {
                        "entry": entry,
                        "matches": matches,
                        "query_score_cosine": max_score,
                        "query_score_weighted": weighted_score,
                        "top_pos_score": top_pos_score,
                        "top_neg_score": top_neg_score,
                        "pos_neg_margin": pos_neg_margin,
                        "positive_count": len(pos_scores),
                        "negative_count": len(neg_scores),
                    }
                )

            rank_scores = choose_ranking_scores(
                chunk_scores_cosine,
                chunk_scores_weighted,
                chunk_scores_margin,
                chunk_scores_max,
                args.ranking_score,
            )
            selected_indices = select_top_n(rank_scores, args.top_n)
            if not selected_indices:
                selected_indices = list(range(len(chunks_with_matches)))

            selected_indices = sorted(selected_indices, key=lambda idx: (-rank_scores[idx], idx))

            for rank, idx in enumerate(selected_indices, start=1):
                item = chunks_with_matches[idx]
                entry = item["entry"]
                hybrid_score = rank_scores[idx]
                entry["query_rank"] = rank
                entry["query_score_cosine"] = item["query_score_cosine"]
                entry["query_score_weighted"] = item["query_score_weighted"]
                entry["top_pos_score"] = item["top_pos_score"]
                entry["top_neg_score"] = item["top_neg_score"]
                entry["pos_neg_margin"] = item["pos_neg_margin"]
                entry["positive_count"] = item["positive_count"]
                entry["negative_count"] = item["negative_count"]
                entry["query_score"] = hybrid_score
                entry["query_rank_score"] = hybrid_score
                entry["query_rank_score_mode"] = args.ranking_score
                entry["index_score_field"] = args.index_score_field
                entry["selected_for_retrieval"] = 1
                selected_records.append(entry)
                selected_query_items.append(
                    {
                        "query_vector_id": entry["vector_id"],
                        "query_chunk_id": entry["chunk_id"],
                        "query_rank": rank,
                        "query_score_cosine": item["query_score_cosine"],
                        "query_score_weighted": item["query_score_weighted"],
                        "top_pos_score": item["top_pos_score"],
                        "top_neg_score": item["top_neg_score"],
                        "pos_neg_margin": item["pos_neg_margin"],
                        "positive_count": item["positive_count"],
                        "negative_count": item["negative_count"],
                        "query_score": hybrid_score,
                        "query_rank_score": hybrid_score,
                        "query_rank_score_mode": args.ranking_score,
                        "index_score_field": args.index_score_field,
                        "index_matches": item["matches"],
                    }
                )

    if args.dry_run:
        with metadata_path.open("w", encoding="utf-8") as f_meta:
            for record in selected_records:
                f_meta.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with metadata_path.open("w", encoding="utf-8") as f_meta:
            for record in selected_records:
                f_meta.write(json.dumps(record, ensure_ascii=False) + "\n")

    query_meta_by_vector_id: Dict[int, Dict] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            vector_id_value = item.get("vector_id")
            if vector_id_value is None:
                continue
            query_meta_by_vector_id[int(vector_id_value)] = item

    if not args.dry_run:
        grouped_results = group_selected_query_items(selected_query_items, query_meta_by_vector_id)
        with results_path.open("w", encoding="utf-8") as f_res_final:
            for grouped_result in grouped_results:
                f_res_final.write(json.dumps(grouped_result, ensure_ascii=False) + "\n")

    print(f"Wrote metadata: {metadata_path}")
    if not args.dry_run:
        print(f"Wrote results: {results_path}")
    print(f"Ranking score mode: {args.ranking_score}")
    print(f"Index score field: {args.index_score_field}")
    print(f"Chunks: {total_chunks}, total lines: {total_lines}, missing idx: {missing_idx}")

    # --- Chunk hit rate evaluation ---
    if args.eval_label_path and not args.dry_run:
        _eval_chunk_hit_rate(results_path, args.eval_label_path, args.top_n)


def _eval_chunk_hit_rate(results_path: Path, label_path: str, top_n: int) -> None:
    """Evaluate how often the selected query chunks contain vulnerability lines.

    Uses the ``labels`` field (0/1 per line) from *label_path* as ground truth
    and the ``chunk_line_ids`` / ``start_line``-``end_line`` from each query chunk to
    determine whether the chunk overlaps with any vulnerability-modified line.

    Prints per-rank and cumulative hit-rate statistics.
    """
    # Load ground-truth line labels
    gt: Dict[int, Dict] = {}
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            labels = rec.get("labels", [])
            target = rec.get("target")
            gt[idx] = {"labels": labels, "target": target}

    # Scan grouped results
    rank_hits: Dict[int, int] = {}
    rank_totals: Dict[int, int] = {}
    cumul_hit = 0
    vuln_func_total = 0
    vuln_func_has_vuln_lines = 0  # how many vuln funcs actually have label=1 lines

    with open(results_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            func_id = rec.get("query_func_id") or rec.get("query_idx")
            target = rec.get("query_target")
            try:
                t = int(target)
            except (TypeError, ValueError):
                continue
            if t != 1:
                continue  # only evaluate on vulnerable functions

            vuln_func_total += 1
            test_labels = gt.get(func_id, {}).get("labels", [])
            if not test_labels:
                continue
            if any(l == 1 for l in test_labels):
                vuln_func_has_vuln_lines += 1

            any_hit = False
            for chunk in rec.get("query_chunks", []):
                rank = chunk.get("query_rank")
                chunk_meta = chunk.get("query_chunk", {})
                line_ids = chunk_meta.get("chunk_line_ids", [])
                if not line_ids:
                    start = chunk_meta.get("start_line", 0)
                    end = chunk_meta.get("end_line", 0)
                    if start and end:
                        line_ids = list(range(start, end + 1))

                has_vuln = any(
                    0 <= (lid - 1) < len(test_labels) and test_labels[lid - 1] == 1
                    for lid in line_ids
                    if isinstance(lid, int)
                )

                rank_totals[rank] = rank_totals.get(rank, 0) + 1
                if has_vuln:
                    rank_hits[rank] = rank_hits.get(rank, 0) + 1
                    if not any_hit:
                        any_hit = True
                        cumul_hit += 1

    # Print report
    print("\n" + "=" * 60)
    print("  Query Chunk Vulnerability-Line Hit Rate Evaluation")
    print("=" * 60)
    print(f"  Vulnerable functions total:          {vuln_func_total}")
    print(f"  With ≥1 label=1 line (achievable):   {vuln_func_has_vuln_lines}/{vuln_func_total} ({vuln_func_has_vuln_lines/vuln_func_total*100:.1f}%)")
    print()

    print(f"  {'Rank':<8} {'Hit':>6} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 35)
    for rank in sorted(rank_totals.keys()):
        hits = rank_hits.get(rank, 0)
        total = rank_totals[rank]
        print(f"  {rank:<8} {hits:>6} {total:>8} {hits/total*100:>7.1f}%")

    print()
    print(f"  Cumulative top-{top_n} hit: {cumul_hit}/{vuln_func_total} ({cumul_hit/vuln_func_total*100:.1f}%)")
    if vuln_func_has_vuln_lines > 0:
        print(f"  Achievable hit rate:     {vuln_func_has_vuln_lines}/{vuln_func_total} ({vuln_func_has_vuln_lines/vuln_func_total*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
