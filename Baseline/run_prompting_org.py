import argparse
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import openai
from openai import OpenAI
from openai._types import NOT_GIVEN

from utils import SYS_INST, PROMPT_INST, PROMPT_INST_COT, ONESHOT_ASSISTANT, ONESHOT_USER, TWOSHOT_USER, TWOSHOT_ASSISTANT
from model_api_clients import (
    get_openai_chat as api_get_openai_chat,
    normalize_usage,
    truncate_tokens_from_messages as api_truncate_tokens_from_messages,
)

import tiktoken

API_KEY = os.getenv("SLICERAG_API_KEY") or os.getenv("OPENAI_API_KEY")
client = (
    OpenAI(
        api_key=API_KEY,
        base_url=os.getenv("SLICERAG_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
    )
    if API_KEY
    else None
)

def truncate_tokens_from_messages(messages, model, max_gen_length):
    return api_truncate_tokens_from_messages(messages, model, max_gen_length)


# get completion from an OpenAI chat model
def build_fewshot_messages(args):
    """Build fixed few-shot messages without editing utils.py between runs."""
    if not args.fewshot_eg or args.fewshot_variant == "none":
        return []

    examples = {
        "yes": [
            {"role": "user", "content": ONESHOT_USER},
            {"role": "assistant", "content": ONESHOT_ASSISTANT},
        ],
        "no": [
            {"role": "user", "content": TWOSHOT_USER},
            {"role": "assistant", "content": TWOSHOT_ASSISTANT},
        ],
    }
    variants = {
        # PrimeVul authors' baseline order: negative example first, then positive.
        "author_no_yes": examples["no"] + examples["yes"],
        # Current utils.py declaration order, useful as an order-sensitivity check.
        "current_yes_no": examples["yes"] + examples["no"],
        "yes_only": examples["yes"],
        "no_only": examples["no"],
    }
    return variants[args.fewshot_variant]


def get_openai_chat(
    prompt,
    args
):
    if args.fewshot_eg:
        messages = [{"role": "system", "content": SYS_INST}]
        messages.extend(build_fewshot_messages(args))
        messages.append({"role": "user", "content": prompt["prompt"]})
    else:
        messages = [
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": prompt["prompt"]}
        ]

    try:
        trunc_messages = api_truncate_tokens_from_messages(messages, args.model, args.max_gen_length)
        response_content, usage, _, reasoning = api_get_openai_chat(
            {"messages": trunc_messages},
            args.model,
            args.prompt_strategy,
            args.temperature,
            args.max_gen_length,
            args.seed,
        )

        return response_content, usage, trunc_messages, reasoning

    except Exception as error:
        print(f"API call failed: {error}")
        return None, {}, messages, None

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        p["idx"] = sample["idx"]
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        p["prompt"] = inst.format(func=sample["func"])
        prompts.append(p)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="glm-5.1", choices=["gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"], help='Model name')
    parser.add_argument('--prompt_strategy', type=str, choices=["std_cls", "cot"], default="std_cls", help='Prompt strategy')
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--output_folder', type=str, help='Output folder')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_gen_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--logprobs', action="store_true", help='Return logprobs')
    parser.add_argument('--fewshot_eg', action="store_true", help='Use few-shot examples')
    parser.add_argument(
        '--fewshot_variant',
        type=str,
        choices=["author_no_yes", "current_yes_no", "yes_only", "no_only", "none"],
        default="author_no_yes",
        help='Fixed few-shot variant. Default matches the authors: NO example, then YES example.',
    )
    parser.add_argument('--num-threads', type=int, default=3, help='Number of worker threads for prompting')
    args = parser.parse_args()

    fewshot_suffix = args.fewshot_variant if args.fewshot_eg else "none"
    output_file = os.path.join(args.output_folder, f"{args.model}_{args.prompt_strategy}_logprobs{args.logprobs}_fewshoteg{args.fewshot_eg}_{fewshot_suffix}.jsonl")
    if args.prompt_strategy == "std_cls":
        inst = PROMPT_INST
    elif args.prompt_strategy == "cot":
        inst = PROMPT_INST_COT
    else:
        raise ValueError("Invalid prompt strategy")
    prompts = construct_prompts(args.data_path, inst)

    def worker(prompt):
        response, usage, messages, reasoning = get_openai_chat(prompt, args)
        usage = normalize_usage(usage, args.model)
        if response is None:
            response = "ERROR"
        return {
            "response": response,
            "usage": usage,
            "messages": messages,
            "reasoning": reasoning,
        }

    max_workers = max(1, args.num_threads)

    with open(output_file, "w") as f:
        print(f"Requesting {args.model} to respond to {len(prompts)} prompts ...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for p, result in tqdm(zip(prompts, executor.map(worker, prompts)), total=len(prompts)):
                p["messages"] = result["messages"]
                p["reasoning"] = result["reasoning"]
                p["usage"] = result["usage"]
                p["response"] = result["response"]
                f.write(json.dumps(p))
                f.write("\n")
                f.flush()


if __name__ == "__main__":
    main()
