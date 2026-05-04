import threading
import time
from typing import Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
)

_model = None
_tokenizer = None
_llm_lock = threading.Lock()

MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
EOS_IDS = {151643, 151645}  # <|endoftext|> and <|im_end|>

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly Chinese language tutor. "
    "Help the user practice conversational Mandarin Chinese. "
    "Keep responses short and natural. "
    "Mix Chinese and English as appropriate for the learner's level."
)


def get_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {MODEL_PATH} on {DEVICE} (4-bit)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_cfg,
            device_map={"": 0},
        )
        _model.eval()
        print("Qwen2.5 3B loaded.")
    return _model, _tokenizer


class _DynamicBiasProcessor(LogitsProcessor):
    """Applies per-token logit bias from ScoresStore; built once per generation call."""

    def __init__(self):
        self._bias: torch.Tensor | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._bias is None:
            from vocabulary import get_qwen_vocab
            from scores import ScoresStore
            all_vocab = get_qwen_vocab()
            store = ScoresStore.get()
            bias_np = store.get_dynamic_bias_array(scores.shape[-1], set(all_vocab.keys()), EOS_IDS)
            self._bias = torch.tensor(bias_np, dtype=torch.float32, device=scores.device)
        return scores + self._bias


def stream_chat(
    user_message: str,
    system_prompt: str | None = None,
    filter_vocab: bool = False,
    produced_token_ids: list[int] | None = None,
) -> Generator[str, None, None]:
    """
    Token-by-token greedy generation with optional dynamic logit bias.
    Mirrors the MLX generate_step loop so logits processors and token capture work identically.
    """
    model, tokenizer = get_model()

    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(DEVICE)

    bias_proc = _DynamicBiasProcessor() if filter_vocab else None
    accumulated: list[int] = []
    decoded_so_far = ""
    past_key_values = None
    current_input = input_ids

    with _llm_lock, torch.no_grad():
        try:
            for _ in range(512):
                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                past_key_values = outputs.past_key_values

                if bias_proc is not None:
                    logits = bias_proc(current_input, logits)

                next_id = int(logits.argmax(-1).item())
                if next_id in EOS_IDS:
                    break

                accumulated.append(next_id)
                if produced_token_ids is not None:
                    produced_token_ids.append(next_id)

                new_decoded = tokenizer.decode(accumulated, skip_special_tokens=True)
                new_text = new_decoded[len(decoded_so_far):]
                decoded_so_far = new_decoded
                if new_text:
                    yield new_text

                current_input = torch.tensor([[next_id]], device=DEVICE)
        finally:
            del past_key_values
            torch.cuda.empty_cache()


def chat(user_message: str, system_prompt: str | None = None) -> tuple[str, float]:
    t0 = time.time()
    tokens = list(stream_chat(user_message, system_prompt))
    elapsed = time.time() - t0
    return "".join(tokens), elapsed
