import os
import subprocess
import tempfile

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import LogitsProcessor, LogitsProcessorList
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

MODEL_NAME = "openai/whisper-small"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VOCAB_BOOST = 3.0  # kept for reference — no longer used for vocab filtering

# Whisper forced prefix: <|zh|>, <|transcribe|>, <|notimestamps|>
# These 3 tokens are forced after BOS, so content starts at generated[4].
_FORCED_PREFIX_LEN = 3

_processor: WhisperProcessor | None = None
_model: WhisperForConditionalGeneration | None = None
_booster_processor: "VocabularyBooster | None" = None


def get_model():
    global _processor, _model
    if _model is None:
        print(f"Loading {MODEL_NAME} on {DEVICE}...")
        _processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        _model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()
        print(f"Whisper loaded on {DEVICE}.")
    return _processor, _model


class LogitsCapturer(LogitsProcessor):
    """Records raw model logits (before any boost) at each decode step."""

    def __init__(self):
        self.captured: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.captured.append(scores[0].detach().cpu())
        return scores  # pass through unchanged


class VocabularyLocker(LogitsProcessor):
    """Hard-mask all non-vocabulary Whisper tokens to -inf.

    Whisper can only emit tokens in the known vocabulary or special tokens
    (EOS, language tag, etc.).  Pronunciation ranks are still computed from
    the pre-mask logits captured by LogitsCapturer, so they reflect true
    recognition difficulty.
    """

    def __init__(self, vocab_token_ids: list[int], special_token_ids: set[int]):
        self._allowed = set(vocab_token_ids) | special_token_ids
        self._mask: torch.Tensor | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._mask is None:
            mask = torch.full((scores.shape[-1],), float("-inf"),
                              dtype=scores.dtype, device=scores.device)
            valid = [i for i in self._allowed if 0 <= i < scores.shape[-1]]
            mask[valid] = 0.0
            self._mask = mask
        return scores + self._mask


def _get_vocab_locker() -> VocabularyLocker:
    global _booster_processor
    if _booster_processor is None:
        processor, _ = get_model()
        from vocabulary import get_whisper_token_ids
        vocab_ids = get_whisper_token_ids()
        special_ids = set(processor.tokenizer.all_special_ids)
        _booster_processor = VocabularyLocker(vocab_ids, special_ids)
        print(f"Whisper vocab locker: {len(vocab_ids)} vocab + {len(special_ids)} special tokens allowed")
    return _booster_processor


def _webm_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode browser webm/opus → 16 kHz mono float32 via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp, "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
            capture_output=True,
            check=True,
        )
        return np.frombuffer(result.stdout, dtype=np.float32)
    finally:
        os.unlink(tmp)


def transcribe(audio_bytes: bytes, boost_vocab: bool = False) -> tuple[str, dict[int, int]]:
    """
    Transcribe audio and return (text, pron_ranks).

    pron_ranks maps {whisper_token_id: rank} for each content token decoded.
    rank=1 means the token was Whisper's top choice (clear pronunciation).
    rank>1 means Whisper had to work harder (and the vocabulary boost may have helped).
    Ranks are computed from pre-boost logits, so they reflect real pronunciation quality.
    """
    processor, model = get_model()
    audio = _webm_to_pcm(audio_bytes)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    # LogitsCapturer must run before VocabularyBooster so it sees pre-boost logits.
    capturer = LogitsCapturer()
    proc_list = LogitsProcessorList([capturer])
    if boost_vocab:
        proc_list.append(_get_vocab_locker())

    generate_kwargs = dict(language="zh", task="transcribe", logits_processor=proc_list)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kwargs)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    # Compute pronunciation ranks from pre-boost logits.
    # predicted_ids[0][0]  = BOS (decoder_start_token_id, not predicted)
    # capturer.captured[i] = logits that produced predicted_ids[0][i+1]
    # Forced prefix occupies positions 1.._FORCED_PREFIX_LEN (steps 0..2), so
    # content tokens start at predicted_ids[0][_FORCED_PREFIX_LEN+1] = index 4.
    pron_ranks: dict[int, int] = {}
    eos_id = processor.tokenizer.eos_token_id
    generated = predicted_ids[0].cpu().tolist()
    content_start = _FORCED_PREFIX_LEN + 1  # index 4

    for j in range(content_start, len(generated)):
        token_id = generated[j]
        if token_id == eos_id:
            break
        cap_idx = j - 1  # capturer.captured[j-1] produced generated[j]
        if cap_idx >= len(capturer.captured):
            break
        raw_logits = capturer.captured[cap_idx]
        rank = int((raw_logits > raw_logits[token_id]).sum().item()) + 1
        # Keep best rank if token appears multiple times in the utterance
        if token_id not in pron_ranks or rank < pron_ranks[token_id]:
            pron_ranks[token_id] = rank

    return text, pron_ranks
