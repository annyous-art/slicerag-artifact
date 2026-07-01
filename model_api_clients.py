import os
import time

import requests
import tiktoken
from anthropic import Anthropic
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


OPENAI_BASE_URL = os.getenv("SLICERAG_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
ANTHROPIC_BASE_URL = os.getenv("SLICERAG_ANTHROPIC_BASE_URL", os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
RESPONSES_API_URL = os.getenv("SLICERAG_RESPONSES_API_URL", f"{OPENAI_BASE_URL.rstrip('/')}/responses")

SUPPORTED_MODELS = {"gpt-5.5", "glm-5.1", "gemini-3.1-pro-preview", "claude-opus-4-7"}


def _require_api_key():
    api_key = os.getenv("SLICERAG_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set SLICERAG_API_KEY or OPENAI_API_KEY before running closed-model API calls.")
    return api_key


def _openai_client():
    return OpenAI(api_key=_require_api_key(), base_url=OPENAI_BASE_URL)


def _anthropic_client():
    return Anthropic(api_key=_require_api_key(), base_url=ANTHROPIC_BASE_URL)


def messages_to_prompt_string(messages):
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            parts.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"Assistant: {msg['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def truncate_tokens_from_messages(messages, model, max_gen_length):
    if model == "gpt-5.5":
        max_tokens = 1050000 - max_gen_length
    elif model == "glm-5.1":
        max_tokens = 200000 - max_gen_length
    elif model == "gemini-3.1-pro-preview":
        max_tokens = 1000000 - max_gen_length
    elif model == "claude-opus-4-7":
        max_tokens = 1000000 - max_gen_length
    else:
        max_tokens = 4096 - max_gen_length

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    num_tokens = 3
    trunc_messages = []

    for message in messages:
        truncated = {}
        num_tokens += tokens_per_message
        for key, value in message.items():
            if not isinstance(value, str):
                truncated[key] = value
                continue

            encoded_value = encoding.encode(value)
            remaining = max_tokens - num_tokens
            if remaining <= 0:
                truncated[key] = "" if key == "content" else value
                break
            if len(encoded_value) > remaining:
                truncated[key] = encoding.decode(encoded_value[:remaining])
                num_tokens = max_tokens
            else:
                truncated[key] = value
                num_tokens += len(encoded_value)
        trunc_messages.append(truncated)

    return trunc_messages


def _raw_usage_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _responses_api_call(messages, model, max_gen_length, prompt_strategy):
    reasoning_effort = "xhigh" if prompt_strategy == "cot" else "none"
    reasoning_summary = "detailed" if prompt_strategy == "cot" else "concise"
    payload = {
        "model": model,
        "input": messages,
        "reasoning": {"effort": reasoning_effort, "summary": reasoning_summary},
        "text": {"verbosity": "low"},
        "max_output_tokens": max(4096, max_gen_length),
    }

    for attempt in range(5):
        try:
            resp = requests.post(
                RESPONSES_API_URL,
                headers={"Authorization": f"Bearer {_require_api_key()}", "Content-Type": "application/json"},
                json=payload,
                timeout=(10, 120),
            )
            if resp.status_code == 200:
                data = resp.json()
                final_answer = "[EMPTY]"
                reasoning_text = None
                for item in data.get("output", []):
                    if isinstance(item, dict) and item.get("type") == "reasoning":
                        for summary_item in item.get("summary", []):
                            if summary_item.get("type") == "summary_text":
                                reasoning_text = summary_item.get("text", "").strip()
                                break
                    elif isinstance(item, dict) and item.get("type") == "message":
                        for content_item in item.get("content", []):
                            if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                                final_answer = content_item.get("text", "").strip()
                return final_answer, data.get("usage", {}), reasoning_text

            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return "[ERROR]", {}, None
        except requests.RequestException:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            return "[ERROR]", {}, None

    return "[ERROR]", {}, None


def _openai_compatible_chat(messages, model, max_gen_length, temperature, prompt_strategy):
    extra_body = None
    request_kwargs = {}
    if model == "glm-5.1":
        extra_body = {"thinking": {"type": "enabled" if prompt_strategy == "cot" else "disabled"}}
    elif model == "gemini-3.1-pro-preview":
        request_kwargs["reasoning_effort"] = "high" if prompt_strategy == "cot" else "low"

    for attempt in range(5):
        try:
            response = _openai_client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_gen_length,
                temperature=temperature,
                extra_body=extra_body,
                **request_kwargs,
            )
            choice = response.choices[0].message
            content = choice.content.strip() if choice.content else "[EMPTY]"
            reasoning = getattr(choice, "reasoning_content", None)
            if reasoning:
                reasoning = reasoning.strip()
            return content, _raw_usage_dict(response), reasoning
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            return "[ERROR]", {}, None
        except BadRequestError:
            return "[ERROR]", {}, None
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return "[ERROR]", {}, None

    return "[ERROR]", {}, None


def _split_system_from_messages(messages):
    system_text = None
    non_system = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            non_system.append(msg)
    return system_text, non_system


def _extract_anthropic_text(content_blocks):
    texts = []
    blocks = content_blocks or []
    if hasattr(blocks, "model_dump"):
        blocks = blocks.model_dump()
    for block in blocks:
        if hasattr(block, "model_dump"):
            block = block.model_dump()
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "output_text") and block.get("text"):
            texts.append(str(block["text"]).strip())
    return "\n".join(texts).strip() or "[EMPTY]"


def _extract_anthropic_thinking(content_blocks):
    blocks = content_blocks or []
    if hasattr(blocks, "model_dump"):
        blocks = blocks.model_dump()
    for block in blocks:
        if hasattr(block, "model_dump"):
            block = block.model_dump()
        if isinstance(block, dict) and block.get("type") == "thinking":
            text = block.get("thinking") or block.get("text")
            if text:
                return str(text).strip()
    return None


def _anthropic_chat(messages, max_gen_length, prompt_strategy):
    system_text, filtered_messages = _split_system_from_messages(messages)
    request = {
        "model": "claude-opus-4-7",
        "max_tokens": max_gen_length,
        "system": system_text,
        "messages": filtered_messages,
    }
    if prompt_strategy == "cot":
        request["thinking"] = {"type": "adaptive", "display": "summarized"}
        request["output_config"] = {"effort": "max"}
    else:
        request["thinking"] = {"type": "disabled"}

    for attempt in range(5):
        try:
            response = _anthropic_client().messages.create(**request)
            data = response.model_dump()
            final_answer = _extract_anthropic_text(data.get("content", []))
            reasoning = _extract_anthropic_thinking(data.get("content", []))
            return final_answer, data.get("usage", {}), reasoning
        except Exception:
            if attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            return "[ERROR]", {}, None

    return "[ERROR]", {}, None


def normalize_usage(usage, model=None):
    if not usage:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": None,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }

    if model and "claude" in model:
        prompt = usage.get("input_tokens", 0)
        completion = usage.get("output_tokens", 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": None,
            "total_tokens": usage.get("total_tokens", prompt + completion),
        }

    prompt = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    completion = usage.get("output_tokens", usage.get("completion_tokens", 0))
    reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": usage.get("total_tokens", prompt + completion),
    }


def get_openai_chat(prompt, model, prompt_strategy, temperature, max_gen_length, seed):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model for this artifact: {model}")

    messages = truncate_tokens_from_messages(prompt["messages"], model, max_gen_length)

    if model == "gpt-5.5":
        response, usage, reasoning = _responses_api_call(messages, model, max_gen_length, prompt_strategy)
    elif model in ("glm-5.1", "gemini-3.1-pro-preview"):
        response, usage, reasoning = _openai_compatible_chat(
            messages, model, max_gen_length, temperature, prompt_strategy
        )
    else:
        response, usage, reasoning = _anthropic_chat(messages, max_gen_length, prompt_strategy)

    return response, usage, messages, reasoning
