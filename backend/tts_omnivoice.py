import io

import numpy as np
import soundfile as sf
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 24000

# OmniVoice voice-design: comma-separated structured tags only (no free-form text).
_VALID_TAGS = {
    "male", "female", "男", "女",
    "child", "teenager", "young adult", "middle-aged", "elderly",
    "儿童", "少年", "青年", "中年", "老年",
    "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch",
    "极低音调", "低音调", "中音调", "高音调", "极高音调",
    "whisper", "耳语",
    "american accent", "british accent", "australian accent", "chinese accent",
    "canadian accent", "indian accent", "korean accent", "portuguese accent",
    "russian accent", "japanese accent",
    "河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话",
    "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
}

# Valid tags: gender (male/female), age (young adult/middle-aged/…),
# pitch (low pitch/moderate pitch/…), accent (chinese accent/…)
# Free-form text like Qwen3-TTS is NOT supported — invalid tags raise ValueError.
_DEFAULT_INSTRUCT = "female, young adult"
_SLOW_INSTRUCT = "female, young adult, low pitch"  # speed param handles actual pace

_model = None


def get_model():
    global _model
    if _model is None:
        from omnivoice import OmniVoice
        torch.cuda.empty_cache()  # free Qwen KV-cache leftovers before loading ~2.3 GB model
        print(f"Loading OmniVoice on {DEVICE}...")
        _model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=DEVICE, dtype=DTYPE
        )
        print("OmniVoice ready.")
    return _model


def speak(
    text: str,
    lang: str = "zh",
    speed: float = 1.0,
    engine: str = "omnivoice",
    slow: bool = False,
    instruct: str | None = None,
    num_step: int = 16,
) -> bytes:
    model = get_model()
    # OmniVoice only accepts structured tags — ignore free-form strings from the frontend
    resolved_instruct = instruct if instruct in _VALID_TAGS else None
    if resolved_instruct is None:
        resolved_instruct = _SLOW_INSTRUCT if slow else _DEFAULT_INSTRUCT
    audio_arrays = model.generate(
        text=text,
        language="Chinese",
        instruct=resolved_instruct,
        speed=0.8 if slow else speed,
        num_step=num_step,
        guidance_scale=2.0,
    )
    audio: np.ndarray = audio_arrays[0]
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def unload():
    global _model
    _model = None
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def speak_stream(
    text: str,
    slow: bool = False,
    instruct: str | None = None,
):
    """Yields 4-byte LE int32 sample rate, then float32 PCM.
    OmniVoice has no streaming API; full audio is generated then returned as one chunk."""
    yield np.array([SAMPLE_RATE], dtype=np.int32).tobytes()
    wav = speak(text, slow=slow, instruct=instruct)
    data, _ = sf.read(io.BytesIO(wav), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    yield data.astype(np.float32).tobytes()
