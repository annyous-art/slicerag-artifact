import argparse
import json
import os

import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_api_clients import get_openai_chat, normalize_usage
from utils_sliced_prompts import SYS_INST, SYS_INST_COT, SYS_INST_FEWSHOT, SYS_INST_COT_FEWSHOT

# Baseline system instruction (from Primevul baseline)
SYS_INST_BASELINE = "You are a security expert that is good at static program analysis."


def truncate_text_by_tokens(text: str, max_tokens: int, encoding) -> str:
    if not text or max_tokens <= 0:
        return text or ""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_match_chunk_label(match):
    label = match.get("index_chunk_label", match.get("chunk_label"))
    if label is None:
        index_chunk = match.get("index_chunk") if isinstance(match.get("index_chunk"), dict) else {}
        label = index_chunk.get("index_chunk_label", index_chunk.get("chunk_label"))
    return safe_int(label)


def get_match_function_label(match):
    label = match.get("function_target")
    if label is None:
        index_chunk = match.get("index_chunk") if isinstance(match.get("index_chunk"), dict) else {}
        label = index_chunk.get("function_target")
    return safe_int(label)


def get_match_evidence_role(match):
    role = match.get("evidence_role")
    if role:
        return str(role)

    chunk_label = get_match_chunk_label(match)
    function_label = get_match_function_label(match)
    if function_label == 1 and chunk_label == 1:
        return "vulnerable_changed"
    if function_label == 0 and chunk_label == 1:
        return "fixed_changed"
    if function_label == 0:
        return "safe_background"
    if function_label == 1:
        return "vulnerable_context"
    return "unknown"


def get_match_evidence_polarity(match):
    polarity = match.get("evidence_polarity")
    if polarity is not None:
        return safe_int(polarity)
    role = get_match_evidence_role(match)
    if role == "vulnerable_changed":
        return 1
    if role in ("fixed_changed", "safe_background"):
        return 0
    return None


def binary_label_text(value):
    if value is None:
        return "UNKNOWN"
    return "YES" if int(value) == 1 else "NO"


def chunk_label_text(value):
    if value is None:
        return "UNKNOWN"
    return "DIFF_CHANGED_REGION" if int(value) == 1 else "UNCHANGED_CONTEXT_REGION"


def evidence_role_text(role):
    mapping = {
        "vulnerable_changed": "VULNERABLE_CHANGED_REGION",
        "fixed_changed": "FIXED_OR_REPAIRED_CHANGED_REGION",
        "safe_background": "SAFE_BACKGROUND_REGION",
        "vulnerable_context": "VULNERABLE_FUNCTION_CONTEXT_REGION",
        "unknown": "UNKNOWN",
    }
    return mapping.get(role or "unknown", "UNKNOWN")


def selected_query_chunks(query_chunks, max_query_chunks):
    query_chunks = [chunk for chunk in query_chunks if isinstance(chunk, dict)]
    query_chunks = sorted(query_chunks, key=lambda item: item.get("query_rank", 0))
    if max_query_chunks and max_query_chunks > 0:
        query_chunks = query_chunks[:max_query_chunks]
    return query_chunks


def selected_index_matches(chunk, max_index_matches):
    index_matches = chunk.get("index_matches") or []
    index_matches = [m for m in index_matches if isinstance(m, dict)]
    index_matches = sorted(index_matches, key=lambda item: item.get("index_rank", 999))
    if max_index_matches and max_index_matches > 0:
        index_matches = index_matches[:max_index_matches]
    return index_matches


def append_evidence_block(lines, title, matches, include_labels=True):
    lines.append(title)
    lines.append("")
    if not matches:
        lines.append("None")
        lines.append("")
        return

    for match_idx, match in enumerate(matches, start=1):
        index_code = match.get("index_code_clean", "")
        score = match.get("faiss_score", match.get("index_score"))
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "UNKNOWN"
        chunk_label = get_match_chunk_label(match)
        function_label = get_match_function_label(match)
        role = get_match_evidence_role(match)

        lines.append(f"Index {match_idx}:")
        lines.append(f"Retrieval Score: {score_text}")
        lines.append("Index Chunk Code:")
        lines.append(index_code or "")
        if include_labels:
            lines.append(f"Index Evidence Role: {evidence_role_text(role)}")
            lines.append(f"Index Diff Label: {chunk_label_text(chunk_label)}")
            lines.append(f"Index Function Label: {binary_label_text(function_label)}")
        lines.append("")



def format_query_result_prompt(
    query_func,
    query_chunks,
    encoding,
    max_query_chunks=1,
    max_index_matches=1,
    evidence_mode="grouped_chunk_label",
):
    query_func = query_func or ""
    query_chunks = selected_query_chunks(query_chunks, max_query_chunks)

    lines = []
    lines.append("Target Function Code:")
    lines.append(query_func or "")
    lines.append("")
    lines.append("Focus Regions:")
    lines.append("")

    for chunk_idx, chunk in enumerate(query_chunks, start=1):
        query_code = chunk.get("query_code_clean", "")
        q_line_ids = chunk.get("chunk_line_ids")
        if not q_line_ids:
            qc = chunk.get("query_chunk") if isinstance(chunk.get("query_chunk"), dict) else {}
            q_line_ids = qc.get("chunk_line_ids", "")
        if isinstance(q_line_ids, (list, tuple)):
            q_line_ids_text = ",".join(str(x) for x in q_line_ids)
        else:
            q_line_ids_text = str(q_line_ids) if q_line_ids else ""

        lines.append(f"Region {chunk_idx}:")
        lines.append("Focus Chunk Code:")
        lines.append(query_code or "")
        lines.append("")

    if evidence_mode == "target_focus":
        return "\n".join(lines)

    all_matches = []
    for chunk in query_chunks:
        index_matches = chunk.get("index_matches") or []
        index_matches = [m for m in index_matches if isinstance(m, dict)]
        all_matches.extend(sorted(index_matches, key=lambda item: item.get("index_rank", 999)))

    label_counts = {"VULNERABLE": 0, "SAFE_OR_FIXED": 0, "UNKNOWN": 0}
    for match in all_matches:
        polarity = get_match_evidence_polarity(match)
        if polarity == 1:
            label_counts["VULNERABLE"] += 1
        elif polarity == 0:
            label_counts["SAFE_OR_FIXED"] += 1
        else:
            label_counts["UNKNOWN"] += 1

    lines.append("Retrieved Evidence Summary:")
    lines.append(
        "Top retrieved evidence roles: "
        f"{label_counts['VULNERABLE']} vulnerable changed, "
        f"{label_counts['SAFE_OR_FIXED']} safe/fixed, "
        f"{label_counts['UNKNOWN']} UNKNOWN."
    )
    lines.append("Retrieved labels are noisy weak references, not ground truth for the target function.")
    lines.append("")

    if evidence_mode in ("grouped_chunk_label", "grouped_no_label"):
        positive_matches = [m for m in all_matches if get_match_evidence_polarity(m) == 1]
        negative_matches = [m for m in all_matches if get_match_evidence_polarity(m) == 0]
        unknown_matches = [m for m in all_matches if get_match_evidence_polarity(m) is None]

        append_evidence_block(
            lines,
            "Retrieved Vulnerable Changed Evidence:",
            positive_matches[:max_index_matches] if max_index_matches and max_index_matches > 0 else positive_matches,
            include_labels=(evidence_mode == "grouped_chunk_label"),
        )
        append_evidence_block(
            lines,
            "Retrieved Fixed/Safe Evidence:",
            negative_matches[:max_index_matches] if max_index_matches and max_index_matches > 0 else negative_matches,
            include_labels=(evidence_mode == "grouped_chunk_label"),
        )
        if unknown_matches:
            append_evidence_block(
                lines,
                "Retrieved Unknown-Label Evidence:",
                unknown_matches[:max_index_matches] if max_index_matches and max_index_matches > 0 else unknown_matches,
                include_labels=False,
            )
        return "\n".join(lines)

    lines.append("Retrieved Evidence:")
    lines.append("")

    for chunk in query_chunks:
        index_matches = selected_index_matches(chunk, max_index_matches)

        for match_idx, match in enumerate(index_matches, start=1):
            index_code = match.get("index_code_clean", "")
            chunk_label = get_match_chunk_label(match)
            function_label = get_match_function_label(match)

            line_ids = match.get("chunk_line_ids")
            if not line_ids:
                ic = match.get("index_chunk") if isinstance(match.get("index_chunk"), dict) else {}
                line_ids = ic.get("chunk_line_ids", "")
            if isinstance(line_ids, (list, tuple)):
                line_ids_text = ",".join(str(x) for x in line_ids)
            else:
                line_ids_text = str(line_ids) if line_ids else ""

            lines.append(f"Index {match_idx}:")
            lines.append("Index Chunk Code:")
            lines.append(index_code or "")
            if evidence_mode == "inline_function_label":
                lines.append(f"Index Label: {binary_label_text(function_label)}")
            elif evidence_mode == "inline_chunk_label":
                lines.append(f"Index Evidence Role: {evidence_role_text(get_match_evidence_role(match))}")
                lines.append(f"Index Diff Label: {chunk_label_text(chunk_label)}")
                lines.append(f"Index Function Label: {binary_label_text(function_label)}")
            elif evidence_mode == "no_label":
                pass
            else:
                lines.append(f"Index Evidence Role: {evidence_role_text(get_match_evidence_role(match))}")
                lines.append(f"Index Diff Label: {chunk_label_text(chunk_label)}")
                lines.append(f"Index Function Label: {binary_label_text(function_label)}")
            lines.append("")

    return "\n".join(lines)


# Few-shot user message template: full-function format, consistent with the baseline
FEWSHOT_USER_TEMPLATE = """Please analyze the following code:
```
{code}
```
Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""

# Two fixed baseline few-shot examples using full functions
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


def format_target_query_prompt(
    query_func,
    query_chunks,
    encoding,
    max_query_chunks=1,
):
    """Generate prompt with only Target Function Code + Focus Regions (no Retrieved Evidence)."""
    query_func = query_func or ""
    query_chunks = [chunk for chunk in query_chunks if isinstance(chunk, dict)]
    query_chunks = sorted(query_chunks, key=lambda item: item.get("query_rank", 0))
    if max_query_chunks and max_query_chunks > 0:
        query_chunks = query_chunks[:max_query_chunks]

    lines = []
    lines.append("Target Function Code:")
    lines.append(query_func or "")
    lines.append("")
    lines.append("Focus Regions:")
    lines.append("")

    for chunk_idx, chunk in enumerate(query_chunks, start=1):
        query_code = chunk.get("query_code_clean", "")
        q_line_ids = chunk.get("chunk_line_ids")
        if not q_line_ids:
            qc = chunk.get("query_chunk") if isinstance(chunk.get("query_chunk"), dict) else {}
            q_line_ids = qc.get("chunk_line_ids", "")
        if isinstance(q_line_ids, (list, tuple)):
            q_line_ids_text = ",".join(str(x) for x in q_line_ids)
        else:
            q_line_ids_text = str(q_line_ids) if q_line_ids else ""

        lines.append(f"Region {chunk_idx}:")
        lines.append("Focus Chunk Code:")
        lines.append(query_code or "")
        lines.append("")

    return "\n".join(lines)


def format_baseline_compatible_context(
    query_chunks,
    encoding,
    max_query_chunks=1,
    max_index_matches=1,
    evidence_mode="target_focus",
):
    """Format RAG context without changing the baseline target-code prompt."""
    query_chunks = selected_query_chunks(query_chunks, max_query_chunks)
    lines = []
    lines.append("Additional reference context:")
    lines.append("The following focus regions and retrieved snippets are automatically selected and may be noisy. Use them only as weak reference.")
    lines.append("")
    lines.append("Focus Regions:")
    lines.append("")

    for chunk_idx, chunk in enumerate(query_chunks, start=1):
        query_code = chunk.get("query_code_clean", "")
        lines.append(f"Region {chunk_idx}:")
        lines.append(query_code or "")
        lines.append("")

    if evidence_mode == "target_focus":
        return "\n".join(lines)

    all_matches = []
    for chunk in query_chunks:
        index_matches = chunk.get("index_matches") or []
        index_matches = [m for m in index_matches if isinstance(m, dict)]
        all_matches.extend(sorted(index_matches, key=lambda item: item.get("index_rank", 999)))

    if evidence_mode in ("grouped_chunk_label", "grouped_no_label"):
        positive_matches = [m for m in all_matches if get_match_evidence_polarity(m) == 1]
        negative_matches = [m for m in all_matches if get_match_evidence_polarity(m) == 0]
        include_labels = evidence_mode == "grouped_chunk_label"
        append_evidence_block(
            lines,
            "Retrieved Vulnerable Changed Evidence:",
            positive_matches[:max_index_matches] if max_index_matches and max_index_matches > 0 else positive_matches,
            include_labels=include_labels,
        )
        append_evidence_block(
            lines,
            "Retrieved Fixed/Safe Evidence:",
            negative_matches[:max_index_matches] if max_index_matches and max_index_matches > 0 else negative_matches,
            include_labels=include_labels,
        )
        return "\n".join(lines)

    lines.append("Retrieved Evidence:")
    lines.append("")
    for chunk in query_chunks:
        for match_idx, match in enumerate(selected_index_matches(chunk, max_index_matches), start=1):
            index_code = match.get("index_code_clean", "")
            lines.append(f"Index {match_idx}:")
            lines.append(index_code or "")
            if evidence_mode == "inline_function_label":
                lines.append(f"Index Label: {binary_label_text(get_match_function_label(match))}")
            elif evidence_mode == "inline_chunk_label":
                lines.append(f"Index Evidence Role: {evidence_role_text(get_match_evidence_role(match))}")
                lines.append(f"Index Diff Label: {chunk_label_text(get_match_chunk_label(match))}")
                lines.append(f"Index Function Label: {binary_label_text(get_match_function_label(match))}")
            elif evidence_mode == "no_label":
                pass
            lines.append("")
    return "\n".join(lines)


def format_baseline_compatible_combined_prompt(
    query_func,
    query_chunks,
    encoding,
    max_query_chunks=1,
    max_index_matches=1,
    evidence_mode="target_focus",
):
    context = format_baseline_compatible_context(
        query_chunks=query_chunks,
        encoding=encoding,
        max_query_chunks=max_query_chunks,
        max_index_matches=max_index_matches,
        evidence_mode=evidence_mode,
    )
    return f"""Please analyze the following code:
```
{query_func or ""}
```

{context}

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Only reply with one of the options above. Do not include any further information.
"""


def extract_index_matches(query_chunks, max_query_chunks=1, max_index_matches=1):
    """Extract and flatten all index_matches from query_chunks for few-shot construction."""
    query_chunks = [chunk for chunk in query_chunks if isinstance(chunk, dict)]
    query_chunks = sorted(query_chunks, key=lambda item: item.get("query_rank", 0))
    if max_query_chunks and max_query_chunks > 0:
        query_chunks = query_chunks[:max_query_chunks]

    all_matches = []
    for chunk in query_chunks:
        index_matches = chunk.get("index_matches") or []
        index_matches = [m for m in index_matches if isinstance(m, dict)]
        if max_index_matches and max_index_matches > 0:
            index_matches = index_matches[:max_index_matches]

        for match in index_matches:
            index_code = match.get("index_code_clean", "")
            label = binary_label_text(get_match_evidence_polarity(match))
            all_matches.append({
                "index_code_clean": index_code,
                "chunk_label": label,
            })

    return all_matches


def append_retrieved_fewshot_messages(messages, query_chunks, max_query_chunks=1, max_index_matches=1):
    """Add retrieved chunks as dialogue-style few-shot examples."""
    for match in extract_index_matches(query_chunks, max_query_chunks, max_index_matches):
        label = match.get("chunk_label", "UNKNOWN")
        if label not in ("YES", "NO"):
            continue
        code = match.get("index_code_clean", "")
        if not code:
            continue
        messages.append({"role": "user", "content": FEWSHOT_USER_TEMPLATE.format(code=code)})
        messages.append({"role": "assistant", "content": label})


def construct_prompts(
    input_file,
    prompt_strategy,
    encoding,
    max_query_chunks=1,
    max_index_matches=1,
    fewshot_eg=False,
    classify_threshold=0,
    evidence_mode="grouped_chunk_label",
    combined_style="rag",
):
    """Construct prompts with an optional length-based classifier.

    When classify_threshold > 0:
      - len(query_func) < threshold  → RAG prompt (SYS_INST + Focus Regions + Retrieved Evidence)
      - len(query_func) >= threshold → Baseline two-shot prompt (SYS_INST_BASELINE + fixed few-shot + code)
    When classify_threshold == 0 (default): original behavior (controlled by --fewshot_eg).
    """
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    prompts = []

    # Pre-select system instructions for each mode
    sys_inst_rag = SYS_INST_COT if prompt_strategy == "cot" else SYS_INST
    sys_inst_rag_fewshot = SYS_INST_COT_FEWSHOT if prompt_strategy == "cot" else SYS_INST_FEWSHOT

    if not samples or not isinstance(samples[0], dict):
        raise ValueError("construct_prompts expects JSONL records")

    if not all("query_chunks" in sample for sample in samples):
        raise ValueError("construct_prompts expects grouped records with query_chunks")

    grouped_results = samples

    # Counters for summary
    rag_count = 0
    baseline_count = 0

    for sample_group in grouped_results:
        query_chunks = sorted(sample_group.get("query_chunks", []), key=lambda item: item.get("query_rank", 0))
        query_func = sample_group.get("query_func", "")
        func_len = len(query_func or "")

        # ----- Classifier: route by function character length -----
        if classify_threshold > 0 and func_len >= classify_threshold:
            # === Baseline two-shot strategy (for long functions) ===
            prompt_mode = "baseline"
            baseline_count += 1
            messages = [{"role": "system", "content": SYS_INST_BASELINE}]
            # Fixed two-shot examples
            messages.append({"role": "user", "content": ONESHOT_USER})
            messages.append({"role": "assistant", "content": ONESHOT_ASSISTANT})
            messages.append({"role": "user", "content": TWOSHOT_USER})
            messages.append({"role": "assistant", "content": TWOSHOT_ASSISTANT})
            # Target function in baseline format (code in backticks)
            user_content = FEWSHOT_USER_TEMPLATE.format(code=query_func)
            messages.append({"role": "user", "content": user_content})

        else:
            # === RAG strategy (for short functions, or when classifier is disabled) ===
            rag_count += 1

            if fewshot_eg and combined_style == "baseline_compatible":
                prompt_mode = f"baseline_compatible_{evidence_mode}"
                messages = [{"role": "system", "content": SYS_INST_BASELINE}]
                # Match the standalone baseline order: NO example, then YES example.
                messages.append({"role": "user", "content": TWOSHOT_USER})
                messages.append({"role": "assistant", "content": TWOSHOT_ASSISTANT})
                messages.append({"role": "user", "content": ONESHOT_USER})
                messages.append({"role": "assistant", "content": ONESHOT_ASSISTANT})
                user_content = format_baseline_compatible_combined_prompt(
                    query_func=query_func,
                    query_chunks=query_chunks,
                    encoding=encoding,
                    max_query_chunks=max_query_chunks,
                    max_index_matches=max_index_matches,
                    evidence_mode=evidence_mode,
                )
                messages.append({"role": "user", "content": user_content})
            else:
                prompt_mode = f"rag_{evidence_mode}"

                if fewshot_eg:
                    sys_inst = sys_inst_rag_fewshot
                else:
                    sys_inst = sys_inst_rag

                messages = [{"role": "system", "content": sys_inst}]

                if fewshot_eg:
                    # Mixed structure: fixed full-function baseline examples plus RAG evidence as inline context
                    messages.append({"role": "user", "content": ONESHOT_USER})
                    messages.append({"role": "assistant", "content": ONESHOT_ASSISTANT})
                    messages.append({"role": "user", "content": TWOSHOT_USER})
                    messages.append({"role": "assistant", "content": TWOSHOT_ASSISTANT})

                if evidence_mode == "retrieved_fewshot":
                    append_retrieved_fewshot_messages(
                        messages,
                        query_chunks=query_chunks,
                        max_query_chunks=max_query_chunks,
                        max_index_matches=max_index_matches,
                    )
                    user_content = format_target_query_prompt(
                        query_func=query_func,
                        query_chunks=query_chunks,
                        encoding=encoding,
                        max_query_chunks=max_query_chunks,
                    )
                else:
                    # RAG user content: Target Function Code + Focus Regions + Retrieved Evidence
                    user_content = format_query_result_prompt(
                        query_func=query_func,
                        query_chunks=query_chunks,
                        encoding=encoding,
                        max_query_chunks=max_query_chunks,
                        max_index_matches=max_index_matches,
                        evidence_mode=evidence_mode,
                    )
                messages.append({"role": "user", "content": user_content})

        # sample_key should come from the query-side project/commit_id
        project = sample_group.get("query_project", "")
        commit_id = sample_group.get("query_commit_id", "")
        if project or commit_id:
            key = f"{project}_{commit_id}"
        else:
            key = str(sample_group.get("query_idx", ""))

        p = {
            "query_idx": sample_group.get("query_idx"),
            "query_func_id": sample_group.get("query_func_id"),
            "query_func": query_func,
            "query_target": sample_group.get("query_target"),
            "query_chunks": query_chunks,
            "messages": messages,
            "sample_key": key,
            "prompt_mode": prompt_mode,
            "func_char_len": func_len,
        }
        prompts.append(p)

    if classify_threshold > 0:
        print(f"[Classifier] threshold={classify_threshold} | RAG: {rag_count}, Baseline: {baseline_count}, Total: {rag_count + baseline_count}")

    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default="glm-5.1",
        choices=["gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"],
    )
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="std_cls", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument(
        '--combined-style',
        type=str,
        choices=["rag", "baseline_compatible"],
        default="rag",
        help='When --fewshot_eg is used, choose original RAG-style combined prompt or baseline-compatible combined prompt.',
    )
    parser.add_argument('--classify-threshold', type=int, default=0, help='Character length threshold for classifier: len(func) < threshold uses RAG prompt, >= threshold uses Baseline two-shot. 0 disables classifier (original behavior).')
    parser.add_argument('--max-query-chunks', type=int, default=1, help='Max query chunks to include per query_func_id in the prompt; 0 means keep all')
    parser.add_argument('--max-index-matches', type=int, default=1, help='Max index chunks to include per query chunk in the prompt; 0 means keep all')
    parser.add_argument(
        '--evidence-mode',
        type=str,
        default="grouped_chunk_label",
        choices=[
            "target_focus",
            "no_label",
            "inline_function_label",
            "inline_chunk_label",
            "grouped_chunk_label",
            "grouped_no_label",
            "retrieved_fewshot",
        ],
        help='RAG evidence formatting/ablation mode.',
    )
    parser.add_argument('--num-threads', type=int, default=3, help='Number of worker threads for prompting')
    args = parser.parse_args()

    try:
        encoding = tiktoken.encoding_for_model(args.model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    output_file = os.path.join(
        args.output_folder,
        f"{args.model}_{args.prompt_strategy}_{args.evidence_mode}_{args.combined_style}_fewshoteg{args.fewshot_eg}_cls{args.classify_threshold}.jsonl",
    )

    prompts = construct_prompts(
        args.data_path,
        prompt_strategy=args.prompt_strategy,
        encoding=encoding,
        max_query_chunks=args.max_query_chunks,
        max_index_matches=args.max_index_matches,
        fewshot_eg=args.fewshot_eg,
        classify_threshold=args.classify_threshold,
        evidence_mode=args.evidence_mode,
        combined_style=args.combined_style,
    )

    def _safe_normalize(usage):
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
        try:
            return normalize_usage(usage, args.model)
        except Exception:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}

    def worker(prompt_item):
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

        if response is None:
            response = "ERROR"

        p = dict(prompt_item)
        p["messages"] = messages
        p["reasoning"] = reasoning
        p["response"] = response
        u = _safe_normalize(usage)
        p["usage"] = u
        p["prompt_tokens"] = u.get("prompt_tokens", 0)
        p["completion_tokens"] = u.get("completion_tokens", 0)
        p["reasoning_tokens"] = u.get("reasoning_tokens", 0)
        p["total_tokens"] = u.get("total_tokens", 0)
        return p

    # Run prompts with a thread pool and write results as they complete.
    with open(output_file, "w", encoding="utf-8") as f:
        if not prompts:
            pass
        else:
            max_workers = max(1, args.num_threads)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(worker, pr): idx for idx, pr in enumerate(prompts)}
                for _f in tqdm(as_completed(futures), total=len(futures), desc="Prompting"):
                    try:
                        result = _f.result()
                    except Exception as e:
                        # fallback record
                        result = {"response": "ERROR", "messages": None, "reasoning": None}
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
