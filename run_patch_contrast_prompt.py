#!/usr/bin/env python3
"""Run patch-aware contrastive RAG prompts.

Input is query_patch_contrast_results.jsonl from build_query_patch_contrast.py.
The prompt presents historical vulnerable/fixed patch pairs and asks the model
whether the target resembles the vulnerable side or the repaired side.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


PATCH_CONTRAST_SYS = """You are a security expert that is good at static program analysis.
You will be given a target function and historical patch pairs. Each historical pair contains:
1. the vulnerable-side changed region before a security fix, and
2. the fixed-side changed region after the repair.

Use the historical pairs as weak comparative evidence. Do not predict YES only because a retrieved pair is security-related. Decide whether the target code behavior more closely preserves the vulnerable-side pattern or the fixed-side repair pattern. If the retrieved pair is irrelevant, ignore it.

Please only reply with one of the following options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability."""


ONESHOT_USER = """Please analyze the following code:
```
int64 ClientUsageTracker::GetCachedHostUsage(const std::string& host) {
   HostUsageMap::const_iterator found = cached_usage_.find(host);
   if (found == cached_usage_.end())
     return 0;

  int64 usage = 0;
  const UsageMap& map = found->second;
  for (UsageMap::const_iterator iter = map.begin();
       iter != map.end(); ++iter) {
    usage += iter->second;
  }
  return usage;
}

```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""
ONESHOT_ASSISTANT = "YES"


TWOSHOT_USER = """Please analyze the following code:
```
static char *clean_path(char *path)
{
        char *ch;
        char *ch2;
        char *str;
        str = xmalloc(strlen(path) + 1);
        ch = path;
        ch2 = str;
        while (true) {
                *ch2 = *ch;
                ch++;
                ch2++;
                if (!*(ch-1))
                        break;
                while (*(ch - 1) == '/' && *ch == '/')
                        ch++;
        }
        /* get rid of trailing / characters */
        while ((ch = strrchr(str, '/'))) {
                if (ch == str)
                        break;
                if (!*(ch+1))
                        *ch = 0;
                else
                        break;
        }
        return str;
}
```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""
TWOSHOT_ASSISTANT = "NO"


def truncate_chars(text: Any, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n/* ... truncated ... */"


def score_text(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "UNKNOWN"


def selected_query_chunks(row: dict, max_query_chunks: int) -> list[dict]:
    chunks = [chunk for chunk in row.get("query_chunks", []) if isinstance(chunk, dict)]
    chunks = sorted(chunks, key=lambda item: item.get("query_rank", 999))
    if max_query_chunks and max_query_chunks > 0:
        chunks = chunks[:max_query_chunks]
    return chunks


def selected_patch_pairs(chunk: dict, max_patch_pairs: int) -> list[dict]:
    pairs = [pair for pair in chunk.get("patch_pair_matches", []) if isinstance(pair, dict)]
    pairs = sorted(pairs, key=lambda item: item.get("patch_pair_rank", 999))
    if max_patch_pairs and max_patch_pairs > 0:
        pairs = pairs[:max_patch_pairs]
    return pairs


def build_user_content(row: dict, args) -> str:
    lines = []
    lines.append("Please analyze the following target function for security vulnerabilities.")
    lines.append("")
    lines.append("Target Function Code:")
    lines.append(truncate_chars(row.get("query_func", ""), args.max_function_chars))
    lines.append("")

    chunks = selected_query_chunks(row, args.max_query_chunks)
    lines.append("Automatically Selected Target Focus Regions:")
    lines.append("These regions are retrieved automatically and may be noisy or incomplete.")
    lines.append("")
    if not chunks:
        lines.append("None")
        lines.append("")
    for chunk_idx, chunk in enumerate(chunks, start=1):
        lines.append(f"Focus Region {chunk_idx}:")
        lines.append(f"Target Lines: {chunk.get('start_line', 'UNKNOWN')}-{chunk.get('end_line', 'UNKNOWN')}")
        lines.append("Target Focus Code:")
        lines.append(truncate_chars(chunk.get("code_clean", ""), args.max_focus_chars))
        lines.append("")

    lines.append("Historical Patch-Aware Contrastive Evidence:")
    lines.append(
        "For each pair, compare the target focus region with both the vulnerable-side changed code and the fixed-side changed code."
    )
    lines.append("")
    if not chunks:
        lines.append("None")
        lines.append("")

    evidence_count = 0
    for chunk_idx, chunk in enumerate(chunks, start=1):
        pairs = selected_patch_pairs(chunk, args.max_patch_pairs)
        for pair_idx, pair in enumerate(pairs, start=1):
            evidence_count += 1
            lines.append(f"Evidence Pair {evidence_count} (for Focus Region {chunk_idx}, rank {pair_idx}):")
            lines.append(f"Project: {pair.get('project', '')}")
            lines.append(f"CWE: {pair.get('cwe', '')}")
            lines.append(f"Target-to-vulnerable score: {score_text(pair.get('vulnerable_side_score'))}")
            lines.append(f"Target-to-fixed score: {score_text(pair.get('fixed_side_score'))}")
            lines.append(f"Contrast margin (vulnerable - fixed): {score_text(pair.get('contrast_margin'))}")
            lines.append(f"Closer historical side: {pair.get('closer_side', 'UNKNOWN')}")
            lines.append("")
            lines.append("Historical vulnerable-side changed region before fix:")
            lines.append(truncate_chars(pair.get("vuln_changed_code", ""), args.max_evidence_chars))
            lines.append("")
            lines.append("Historical fixed-side changed region after fix:")
            lines.append(truncate_chars(pair.get("fixed_changed_code", ""), args.max_evidence_chars))
            lines.append("")

    if evidence_count == 0:
        lines.append("None")
        lines.append("")

    lines.append("Decision rule:")
    lines.append("- Answer YES if the target appears to preserve the vulnerable-side behavior or misses the kind of guard/validation/lifetime repair shown in the fixed side.")
    lines.append("- Answer NO if the target already has the relevant fixed-side protection or if the historical evidence is not behaviorally relevant.")
    lines.append("- Base the final decision on the target function code, not on the retrieval scores alone.")
    lines.append("")
    lines.append("Please only output YES or NO.")
    return "\n".join(lines)


def construct_prompts(args) -> list[dict]:
    prompts = []
    with Path(args.data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = [{"role": "system", "content": PATCH_CONTRAST_SYS}]
            if args.fewshot_eg:
                # Match the stronger standalone baseline order: NO example, then YES example.
                messages.append({"role": "user", "content": TWOSHOT_USER})
                messages.append({"role": "assistant", "content": TWOSHOT_ASSISTANT})
                messages.append({"role": "user", "content": ONESHOT_USER})
                messages.append({"role": "assistant", "content": ONESHOT_ASSISTANT})
            messages.append({"role": "user", "content": build_user_content(row, args)})
            prompts.append(
                {
                    "query_idx": row.get("query_idx"),
                    "query_func_id": row.get("query_func_id"),
                    "query_func": row.get("query_func", ""),
                    "query_target": row.get("query_target"),
                    "project": row.get("project", ""),
                    "cwe": row.get("cwe", ""),
                    "query_chunks": row.get("query_chunks", []),
                    "messages": messages,
                    "prompt_mode": "patch_contrast_fewshot" if args.fewshot_eg else "patch_contrast",
                }
            )
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patch-aware contrastive RAG prompt.")
    parser.add_argument("--data-path", required=True, help="query_patch_contrast_results.jsonl")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument(
        "--model",
        default="glm-5.1",
        choices=["gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"],
    )
    parser.add_argument("--prompt_strategy", choices=["std_cls", "cot"], default="std_cls")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_gen_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fewshot_eg", action="store_true")
    parser.add_argument("--max-query-chunks", type=int, default=1)
    parser.add_argument("--max-patch-pairs", type=int, default=2)
    parser.add_argument("--max-function-chars", type=int, default=50000)
    parser.add_argument("--max-focus-chars", type=int, default=4000)
    parser.add_argument("--max-evidence-chars", type=int, default=3000)
    parser.add_argument("--num-threads", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Write prompt records without calling the model.")
    args = parser.parse_args()

    if not args.dry_run:
        from model_api_clients import get_openai_chat, normalize_usage
    else:
        get_openai_chat = None

        def normalize_usage(_usage, _model):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    os.makedirs(args.output_folder, exist_ok=True)
    output_file = os.path.join(
        args.output_folder,
        f"{args.model}_{args.prompt_strategy}_patch_contrast_fewshoteg{args.fewshot_eg}.jsonl",
    )
    prompts = construct_prompts(args)

    def _safe_normalize(usage):
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        try:
            return normalize_usage(usage, args.model)
        except Exception:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    def worker(prompt_item):
        if args.dry_run:
            row = dict(prompt_item)
            row["response"] = "DRY_RUN"
            row["usage"] = _safe_normalize(None)
            return row
        try:
            response, usage, messages, reasoning = get_openai_chat(
                prompt_item,
                args.model,
                args.prompt_strategy,
                args.temperature,
                args.max_gen_length,
                args.seed,
            )
        except Exception:
            response, usage, messages, reasoning = None, None, None, None
        row = dict(prompt_item)
        row["messages"] = messages or prompt_item.get("messages")
        row["reasoning"] = reasoning
        row["response"] = response or "ERROR"
        usage_row = _safe_normalize(usage)
        row["usage"] = usage_row
        row["prompt_tokens"] = usage_row.get("prompt_tokens", 0)
        row["completion_tokens"] = usage_row.get("completion_tokens", 0)
        row["reasoning_tokens"] = usage_row.get("reasoning_tokens", 0)
        row["total_tokens"] = usage_row.get("total_tokens", 0)
        return row

    with open(output_file, "w", encoding="utf-8") as handle:
        max_workers = max(1, args.num_threads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, prompt): idx for idx, prompt in enumerate(prompts)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prompting"):
                try:
                    result = future.result()
                except Exception:
                    result = {"response": "ERROR", "messages": None, "reasoning": None}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

    print(output_file)


if __name__ == "__main__":
    main()
