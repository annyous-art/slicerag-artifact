#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Tuple

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from chunk_scoring import augment_chunk_metadata


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


def get_evidence_role(function_target, chunk_label: int) -> str:
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


def get_evidence_polarity(evidence_role: str):
    if evidence_role == "vulnerable_changed":
        return 1
    if evidence_role in ("fixed_changed", "safe_background"):
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
    function_target = sample.get("target", "")
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
        evidence_role = get_evidence_role(function_target, chunk_label)
        evidence_polarity = get_evidence_polarity(evidence_role)

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
            "function_id": idx,
            "function_target": function_target,
            **({"is_fixed_version": is_fixed_version} if has_is_fixed_version else {}),
            "chunk_seq": local_chunk_id,
            "start_line": start_raw + 1,
            "end_line": end_raw + 1,
            "raw_line_ids": raw_line_ids,
            "code_raw": "\n".join(raw_lines),
            "code_clean": "\n".join(chunk_valid_lines),  # valid code lines only
            "chunk_line_ids": chunk_line_ids,
            "line_labels": chunk_labels,
            "chunk_label": chunk_label,
            "evidence_role": evidence_role,
            "evidence_polarity": evidence_polarity,
            "is_comment": comment_ids,
            "is_blank": blank_ids,
            "is_brace": brace_ids,
            "valid_count": len(chunk_valid_lines),
        })
        chunk_id += 1
        local_chunk_id += 1

    return chunks_meta, chunk_id, len(lines)


def encode_chunks(texts: List[str], tokenizer, model, device: str, batch_size: int, max_length: int, model_type: str = 'auto') -> "List[List[float]]":
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
            # Decide whether to call model.encoder (seq2seq encoders like CodeT5)
            # or call model(...) directly (BERT/CodeBERT/GraphCodeBERT/Roberta-like).
            if model_type != 'auto':
                use_encoder = (model_type == 'codet5')
            else:
                use_encoder = bool(getattr(getattr(model, 'config', None), 'is_encoder_decoder', False))

            if use_encoder:
                # Seq2Seq encoder: accepts input_ids, attention_mask
                encoder_outputs = model.encoder(**tokens)
                last_hidden = encoder_outputs.last_hidden_state
            else:
                # Encoder-less call: use the model forward which handles embeddings
                outputs = model(**tokens)
                # most transformer models return last_hidden_state
                last_hidden = getattr(outputs, 'last_hidden_state', None)
                if last_hidden is None and isinstance(outputs, tuple) and len(outputs) > 0:
                    last_hidden = outputs[0]
            attention_mask = tokens["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = (last_hidden * mask).sum(1)
            sum_mask = mask.sum(1).clamp(min=1e-6)
            emb = sum_embeddings / sum_mask
            embeddings.extend(emb.cpu().numpy().tolist())

    return embeddings


def build_faiss_index(vectors: List[List[float]], use_ip: bool):
    import numpy as np
    import faiss

    mat = np.asarray(vectors, dtype="float32")
    if use_ip:
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
    else:
        index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    return index


def init_faiss_index(dim: int, use_ip: bool):
    import faiss

    if use_ip:
        return faiss.IndexFlatIP(dim)
    return faiss.IndexFlatL2(dim)



def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk index and metadata from labeled JSONL")
    parser.add_argument("--input", required=True, help="Path to labeled JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory for index and metadata")
    parser.add_argument(
        "--model",
        default=os.getenv("SLICERAG_EMBED_MODEL", "microsoft/graphcodebert-base"),
        help="Embedding model name or path. Override with SLICERAG_EMBED_MODEL for local checkpoints.",
    )
    parser.add_argument("--model-type", choices=["auto", "codet5", "codebert", "graphcodebert"], default="graphcodebert", help="Model type to control encoding (auto detects by model attributes)")
    parser.add_argument("--chunk-size", type=int, default=6, help="Chunk size")
    parser.add_argument("--stride", type=int, default=2, help="Chunk stride")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max input tokens kept for embedding; this truncates the encoder input, not generation output")
    parser.add_argument("--device", default="", help="Device for model, e.g. cpu or cuda")
    parser.add_argument("--use-ip", action="store_true", help="Use inner product for FAISS index")
    parser.add_argument("--dry-run", action="store_true", help="Skip model/FAISS; only write metadata")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars (tqdm)")
    parser.add_argument(
        "--keyword-file",
        default=str(Path(__file__).resolve().parent.parent / "data" / "merged_keywords.txt"),
        help="Path to the keyword list used for chunk keyword scoring",
    )
    parser.add_argument("--tfidf-weight", type=float, default=1.0, help="Weight for normalized TF-IDF score")
    parser.add_argument("--keyword-weight", type=float, default=1.0, help="Weight for normalized keyword score")
    parser.add_argument("--label-weight", type=float, default=1.0, help="Weight for normalized label score")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "chunk_metadata.jsonl"
    index_path = output_dir / "chunk_index.faiss"

    chunk_id = 0
    vector_id = 0
    total_lines = 0
    missing_idx = 0
    total_chunks = 0
    indexed_chunks = 0

    metadata_path = output_dir / "chunk_metadata.jsonl"
    index_path = output_dir / "chunk_index.faiss"

    tokenizer = None
    model = None
    index = None
    texts_batch: List[str] = []
    show_progress = not args.no_progress
    pbar_samples = None
    pbar_chunks = None
    pbar_embeddings = None
    pbar_positive = None

    with input_path.open("r", encoding="utf-8") as f_in, metadata_path.open("w", encoding="utf-8") as f_out:
        if show_progress:
            try:
                from tqdm import tqdm
                pbar_samples = tqdm(desc="Samples", unit="lines")
                pbar_chunks = tqdm(desc="Chunks", unit="chunks")
                pbar_embeddings = tqdm(desc="Embeddings", unit="vecs")
            except Exception:
                print("tqdm not available, continuing without progress bars")
        try:
            for line in f_in:
                line = line.strip()
                if pbar_samples is not None:
                    pbar_samples.update(1)
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
                for entry in new_chunks:
                    entry["vector_id"] = vector_id
                    vector_id += 1
                    total_chunks += 1
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if pbar_chunks is not None:
                        pbar_chunks.update(1)
                    # Index both positive and negative chunks. chunk_label is kept in metadata
                    # so retrieval can separate positive and negative evidence later.
                    if not args.dry_run:
                        texts_batch.append(entry["code_clean"])
                        indexed_chunks += 1

                if not args.dry_run and len(texts_batch) >= args.batch_size:
                    if tokenizer is None or model is None:
                        from transformers import AutoModel, AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(args.model)
                        model = AutoModel.from_pretrained(args.model)
                        model.eval()
                        if args.device:
                            model.to(args.device)

                    vectors = encode_chunks(texts_batch, tokenizer, model, args.device, args.batch_size, args.max_length, model_type=args.model_type)
                    import numpy as np
                    import faiss
                    mat = np.asarray(vectors, dtype="float32")
                    if index is None:
                        index = init_faiss_index(mat.shape[1], args.use_ip)
                    elif index.d != mat.shape[1]:
                        raise ValueError(f"Embedding dim mismatch: index {index.d} vs batch {mat.shape[1]}")
                    if args.use_ip:
                        faiss.normalize_L2(mat)
                    index.add(mat)
                    if pbar_embeddings is not None:
                        try:
                            pbar_embeddings.update(mat.shape[0])
                        except Exception:
                            pass
                    texts_batch = []
        finally:
            if pbar_samples is not None:
                pbar_samples.close()
            if pbar_chunks is not None:
                pbar_chunks.close()
            if pbar_embeddings is not None:
                pbar_embeddings.close()

    if total_chunks == 0:
        print("No chunks produced. Check input or chunk size.")
        return

    if not args.dry_run and indexed_chunks > 0:
        if texts_batch:
            if tokenizer is None or model is None:
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                model = AutoModel.from_pretrained(args.model)
                model.eval()
                if args.device:
                    model.to(args.device)

            vectors = encode_chunks(texts_batch, tokenizer, model, args.device, args.batch_size, args.max_length, model_type=args.model_type)
            import numpy as np
            import faiss
            mat = np.asarray(vectors, dtype="float32")
            if index is None:
                index = init_faiss_index(mat.shape[1], args.use_ip)
            elif index.d != mat.shape[1]:
                raise ValueError(f"Embedding dim mismatch: index {index.d} vs batch {mat.shape[1]}")
            if args.use_ip:
                faiss.normalize_L2(mat)
            index.add(mat)

        import faiss
        faiss.write_index(index, str(index_path))

        if index.ntotal != indexed_chunks:
            print(f"ERROR: index size mismatch: index.ntotal={index.ntotal}, indexed_chunks={indexed_chunks}")
            sys.exit(1)

    augment_chunk_metadata(
        metadata_path,
        args.keyword_file,
        tfidf_weight=args.tfidf_weight,
        keyword_weight=args.keyword_weight,
        label_weight=args.label_weight,
    )
    # Re-read augmented metadata (now contains score_combined). Keep both positive
    # and negative chunks in metadata; chunk_label distinguishes evidence type.
    with metadata_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if indexed_chunks == 0:
        print(f"Wrote metadata: {metadata_path}")
        print(f"Chunks: {total_chunks}, indexed chunks: {indexed_chunks}, total lines: {total_lines}, missing idx: {missing_idx}")
        return

    if args.dry_run:
        print(f"Wrote metadata: {metadata_path}")
        print(f"Chunks: {total_chunks}, total lines: {total_lines}, missing idx: {missing_idx}")
        print("Dry-run: skipped FAISS index building")
        return

    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote FAISS index: {index_path}")
    print(f"Chunks: {total_chunks}, indexed chunks: {indexed_chunks}, total lines: {total_lines}, missing idx: {missing_idx}")


if __name__ == "__main__":
    main()
