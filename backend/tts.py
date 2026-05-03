import io

import numpy as np
import soundfile as sf

# ── Kokoro (PyTorch) ──────────────────────────────────────────────────────────
_kokoro_zh = None
_kokoro_en = None


def _get_kokoro(lang: str):
    global _kokoro_zh, _kokoro_en
    if lang == "zh":
        if _kokoro_zh is None:
            from kokoro import KPipeline
            print("Loading Kokoro ZH…")
            _kokoro_zh = KPipeline(lang_code="z")
            print("Kokoro ZH ready.")
        return _kokoro_zh, "zf_xiaobei"
    else:
        if _kokoro_en is None:
            from kokoro import KPipeline
            print("Loading Kokoro EN…")
            _kokoro_en = KPipeline(lang_code="a")
            print("Kokoro EN ready.")
        return _kokoro_en, "af_heart"


def _speak_kokoro(text: str, lang: str = "zh", speed: float = 1.0) -> bytes:
    pipeline, voice = _get_kokoro(lang)
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            chunks.append(audio)
    if not chunks:
        return b""
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
    return buf.getvalue()


# ── Kokoro MLX (Apple Silicon native) ────────────────────────────────────────
_kokoro_mlx_model = None

_KOKORO_MLX_VOICES = {"zh": "zf_xiaobei", "en": "af_heart"}
_KOKORO_MLX_LANG_CODES = {"zh": "z", "en": "a"}


def _get_kokoro_mlx():
    global _kokoro_mlx_model
    if _kokoro_mlx_model is None:
        from mlx_audio.tts.utils import load_model
        print("Loading Kokoro MLX (mlx-community/Kokoro-82M-bf16)…")
        _kokoro_mlx_model = load_model("mlx-community/Kokoro-82M-bf16")
        print("Kokoro MLX ready.")
    return _kokoro_mlx_model


def _speak_kokoro_mlx(text: str, lang: str = "zh", speed: float = 1.0) -> bytes:
    model = _get_kokoro_mlx()
    voice = _KOKORO_MLX_VOICES.get(lang, "zf_xiaobei")
    lang_code = _KOKORO_MLX_LANG_CODES.get(lang, "z")
    chunks = []
    for result in model.generate(text=text, voice=voice, lang_code=lang_code, speed=speed):
        if result is not None and result.audio is not None:
            chunks.append(np.array(result.audio))
    if not chunks:
        return b""
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
    return buf.getvalue()


# ── Qwen3-TTS CustomVoice (MLX, Apple Silicon native) ────────────────────────
# Available speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric (Chinese)
#                     Ryan, Aiden (English), Ono_Anna (Japanese), Sohee (Korean)
_qwen3_tts_model = None

_QWEN3_VOICE_DEFAULT = "Vivian"
_QWEN3_STYLE_SLOW = "说话慢一点，发音清晰，语气耐心"


def _get_qwen3_tts():
    global _qwen3_tts_model
    if _qwen3_tts_model is None:
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:
            raise RuntimeError(
                "mlx-audio not installed. Run: pip install -U mlx-audio"
            ) from e
        print("Loading Qwen3-TTS (0.6B CustomVoice 4-bit)…")
        _qwen3_tts_model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
        print("Qwen3-TTS ready.")
    return _qwen3_tts_model


def _speak_qwen3_tts(text: str, slow: bool = False, instruct: str | None = None) -> bytes:
    model = _get_qwen3_tts()
    style = instruct if instruct else (_QWEN3_STYLE_SLOW if slow else None)
    chunks = []
    for result in model.generate(text=text, voice=_QWEN3_VOICE_DEFAULT, instruct=style):
        if result.audio is not None:
            chunks.append(np.array(result.audio))
    if not chunks:
        return b""
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
    return buf.getvalue()


def speak_stream_qwen3_tts(
    text: str,
    slow: bool = False,
    instruct: str | None = None,
):
    """Generator: yields 4-byte LE int32 sample rate, then float32 PCM chunks."""
    SAMPLE_RATE = 24000
    yield np.array([SAMPLE_RATE], dtype=np.int32).tobytes()
    model = _get_qwen3_tts()
    style = instruct if instruct else (_QWEN3_STYLE_SLOW if slow else None)
    for result in model.generate(
        text=text,
        voice=_QWEN3_VOICE_DEFAULT,
        instruct=style,
        stream=True,
        streaming_interval=0.5,
    ):
        if result.audio is not None:
            yield np.array(result.audio, dtype=np.float32).tobytes()


def warmup_qwen3_tts() -> None:
    print("Warming up Qwen3-TTS JIT…")
    _speak_qwen3_tts("你好")
    print("Qwen3-TTS warm-up done.")


# ── Public API ────────────────────────────────────────────────────────────────

def speak(
    text: str,
    lang: str = "zh",
    speed: float = 1.0,
    engine: str = "kokoro",
    slow: bool = False,
    instruct: str | None = None,
) -> bytes:
    if engine == "qwen3-tts":
        return _speak_qwen3_tts(text, slow=slow, instruct=instruct)
    if engine == "kokoro-mlx":
        return _speak_kokoro_mlx(text, lang, speed=speed if not slow else 0.75)
    return _speak_kokoro(text, lang, speed=speed if not slow else 0.75)
