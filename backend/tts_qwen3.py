import gc
import io

import numpy as np
import soundfile as sf
import torch

DEVICE = "cuda:0"
SPEAKER = "Vivian"  # Chinese female; alternatives: Serena, Eric, Dylan (CN), Ryan (EN)

_model = None


def get_model():
    global _model
    if _model is None:
        from qwen_tts import Qwen3TTSModel
        torch.cuda.empty_cache()
        print("Loading Qwen3-TTS-12Hz-0.6B-CustomVoice fp16...")
        _model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map=DEVICE,
            dtype=torch.float16,
        )
        print("Qwen3-TTS ready.")
    return _model


def speak(
    text: str,
    lang: str = "zh",
    speed: float = 1.0,
    engine: str = "qwen3-tts",
    slow: bool = False,
    instruct: str | None = None,
) -> bytes:
    model = get_model()
    effective_instruct = instruct if instruct else ("慢速，发音清晰" if slow else None)
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=SPEAKER,
        language="Chinese",
        instruct=effective_instruct,
        non_streaming_mode=True,
    )
    audio: np.ndarray = wavs[0]
    torch.cuda.empty_cache()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()
