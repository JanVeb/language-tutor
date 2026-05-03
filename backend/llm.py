import threading
import time
from typing import Generator

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step

_model = None
_tokenizer = None
_llm_lock = threading.Lock()  # serialise concurrent LLM calls (translate vs. stream)

MODEL_PATH = "mlx-community/Qwen2.5-3B-Instruct-4bit"
EOS_IDS = {151643, 151645}  # <|endoftext|> and <|im_end|>

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly Chinese language tutor. "
    "Help the user practice conversational Mandarin Chinese. "
    "Keep responses short and natural. "
    "Mix Chinese and English as appropriate for the learner's level."
)


def get_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading model {MODEL_PATH}...")
        _model, _tokenizer = load(MODEL_PATH)
        print("Model loaded.")
    return _model, _tokenizer


def _build_dynamic_bias(vocab_size: int) -> mx.array:
    """
    Build per-token logit bias for one generation turn using ScoresStore.
    Returns an mx.array of shape [vocab_size]:
      -inf for non-vocab tokens, scores-driven float for vocab tokens, 0 for EOS.
    """
    from vocabulary import get_qwen_vocab
    from scores import ScoresStore
    all_vocab = get_qwen_vocab()  # {qwen_token_id: text}
    store = ScoresStore.get()
    bias_np = store.get_dynamic_bias_array(vocab_size, set(all_vocab.keys()), EOS_IDS)
    return mx.array(bias_np)


def stream_chat(
    user_message: str,
    system_prompt: str | None = None,
    filter_vocab: bool = False,
    produced_token_ids: list[int] | None = None,
) -> Generator[str, None, None]:
    """
    Stream response tokens as text.
    If produced_token_ids is provided, each generated Qwen token ID is appended to it
    (used by the caller for post-turn score updates).
    """
    model, tokenizer = get_model()

    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = mx.array(tokenizer.encode(prompt_text))

    logits_processors = None
    if filter_vocab:
        bias_ref: list[mx.array] = []
        def _dynamic_bias_filter(tokens: mx.array, logits: mx.array) -> mx.array:
            if not bias_ref:
                bias_ref.append(_build_dynamic_bias(logits.shape[-1]))
            return logits + bias_ref[0]
        logits_processors = [_dynamic_bias_filter]

    accumulated: list[int] = []
    decoded_so_far = ""

    with _llm_lock:
        for token, _ in generate_step(
            prompt_tokens, model,
            max_tokens=512,
            logits_processors=logits_processors,
        ):
            token_id = int(token)
            if token_id in EOS_IDS:
                break
            accumulated.append(token_id)
            if produced_token_ids is not None:
                produced_token_ids.append(token_id)
            new_decoded = tokenizer.decode(accumulated)
            new_text = new_decoded[len(decoded_so_far):]
            decoded_so_far = new_decoded
            if new_text:
                yield new_text


def chat(user_message: str, system_prompt: str | None = None) -> tuple[str, float]:
    t0 = time.time()
    tokens = list(stream_chat(user_message, system_prompt))
    elapsed = time.time() - t0
    return "".join(tokens), elapsed
