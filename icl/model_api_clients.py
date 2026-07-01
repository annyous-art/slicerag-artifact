import importlib.util
from pathlib import Path


_ROOT_CLIENT = Path(__file__).resolve().parents[1] / "model_api_clients.py"
_SPEC = importlib.util.spec_from_file_location("_slicerag_model_api_clients", _ROOT_CLIENT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

get_openai_chat = _MODULE.get_openai_chat
messages_to_prompt_string = _MODULE.messages_to_prompt_string
normalize_usage = _MODULE.normalize_usage
truncate_tokens_from_messages = _MODULE.truncate_tokens_from_messages
