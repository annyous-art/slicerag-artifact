import argparse
import concurrent.futures
import json
import os

from tqdm import tqdm

from utils import PROMPT_INST, PROMPT_INST_COT, SYS_INST, SYS_INST_COT


def normalize_label(value, target=None):
    if isinstance(value, str):
        label = value.strip().upper()
        if label.startswith("YES") or label.startswith("(1)") or label == "1":
            return "YES"
        if label.startswith("NO") or label.startswith("(2)") or label == "2":
            return "NO"
    if target == 1:
        return "YES"
    if target == 0:
        return "NO"
    return None


def construct_prompts(input_file, prompt_strategy, max_examples=2, use_fewshot=True):
    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    prompts = []
    sys_inst = SYS_INST_COT if prompt_strategy == "cot" else SYS_INST
    prompt_inst = PROMPT_INST_COT if prompt_strategy == "cot" else PROMPT_INST

    for sample in samples:
        messages = [{"role": "system", "content": sys_inst}]
        similar_list = sample.get("similar") or []
        examples_used = 0

        if use_fewshot:
            for sim in similar_list[:max_examples]:
                if not isinstance(sim, dict):
                    continue

                sim_func = sim.get("func", "")
                sim_target = sim.get("target", None)
                sim_label = normalize_label(sim.get("Assistant"), sim_target)

                if not sim_label or not isinstance(sim_func, str) or not sim_func.strip():
                    continue

                user_sim = prompt_inst.format(func=sim_func)
                assistant_content = f"ANSWER: {sim_label}" if prompt_strategy == "cot" else sim_label

                messages.append({"role": "user", "content": user_sim})
                messages.append({"role": "assistant", "content": assistant_content})
                examples_used += 1

        target_func = sample.get("func", "")
        target_user = prompt_inst.format(func=target_func)
        messages.append({"role": "user", "content": target_user})

        prompts.append(
            {
                "idx": sample["idx"],
                "sample_key": f"{sample.get('project', '')}_{sample.get('commit_id', '')}",
                "func": sample.get("func"),
                "target": sample.get("target"),
                "similar": similar_list,
                "num_similar_available": len(similar_list),
                "num_examples_used": examples_used,
                "max_examples": max_examples,
                "prompt_format": "baseline_compatible_retrieved_fewshot",
                "truncation": "none_in_prompt_construction",
                "messages": messages,
            }
        )

    return prompts


def get_openai_chat(prompt, args):
    from model_api_clients import get_openai_chat as api_get_openai_chat

    response_content, usage, messages, reasoning = api_get_openai_chat(
        prompt,
        args.model,
        args.prompt_strategy,
        args.temperature,
        args.max_gen_length,
        args.seed,
    )
    return response_content, usage, messages, reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="glm-5.1",
        choices=[
            "gpt-5.5",
            "glm-5.1",
            "gemini-3.1-pro-preview",
            "claude-opus-4-7",
        ],
    )
    parser.add_argument("--prompt_strategy", type=str, choices=["std_cls", "cot"], default="std_cls")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_gen_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fewshot_eg", action="store_true")
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=3, help="Number of worker threads for prompting")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    output_file = os.path.join(
        args.output_folder,
        (
            f"{args.model}_{args.prompt_strategy}_fewshoteg{args.fewshot_eg}"
            f"_top{args.max_examples}_baseline_compatible.jsonl"
        ),
    )

    prompts = construct_prompts(
        args.data_path,
        prompt_strategy=args.prompt_strategy,
        max_examples=args.max_examples,
        use_fewshot=args.fewshot_eg,
    )

    def worker(prompt):
        from model_api_clients import normalize_usage

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

    with open(output_file, "w", encoding="utf-8") as f_out:
        print(f"Requesting {args.model} to respond to {len(prompts)} {args.data_path} prompts ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [(executor.submit(worker, prompt), prompt) for prompt in prompts]

            for future, prompt in tqdm(futures, total=len(futures)):
                try:
                    result = future.result()
                    prompt["messages"] = result["messages"]
                    prompt["reasoning"] = result["reasoning"]
                    prompt["usage"] = result["usage"]
                    prompt["response"] = result["response"]
                except Exception as exc:
                    print(f"Error processing prompt {prompt.get('idx', 'unknown')}: {exc}")
                    prompt["response"] = "ERROR"
                    prompt["usage"] = None
                    prompt["reasoning"] = None

                f_out.write(json.dumps(prompt, ensure_ascii=False) + "\n")
                f_out.flush()


if __name__ == "__main__":
    main()
