"""
Qwen3-TTS standalone server — port 8001.
Must run in venv_qwen3 (isolated from main venv due to transformers version conflict).

Start:
    venv_qwen3/bin/uvicorn tts_qwen3_server:app --host 0.0.0.0 --port 8001
"""

import gc
import io

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title="Qwen3-TTS server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEVICE = "cuda:0"
SPEAKER = "Vivian"      # female Chinese speaker; alternatives: Serena, Eric, Dylan (CN), Ryan (EN)
SAMPLE_RATE = 24000     # set after first load

_model = None


def get_model():
    global _model
    if _model is None:
        from qwen_tts import Qwen3TTSModel
        torch.cuda.empty_cache()
        print(f"Loading {MODEL_ID} on {DEVICE} fp16...")
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=torch.float16,
        )
        print("Qwen3-TTS ready.")
    return _model


@app.post("/speak")
async def speak(
    text: str,
    slow: bool = False,
    instruct: str | None = None,
):
    import asyncio

    def _generate():
        model = get_model()
        effective_instruct = instruct if instruct else ("慢速，发音清晰" if slow else None)
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=SPEAKER,
            language="Chinese",
            instruct=effective_instruct,
            non_streaming_mode=True,
        )
        return wavs[0], sr

    audio, sr = await asyncio.to_thread(_generate)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/unload")
def unload():
    global _model
    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "unloaded"}


@app.get("/health")
def health():
    loaded = _model is not None
    mem_mb = round(torch.cuda.memory_allocated(0) / 1e6, 1) if torch.cuda.is_available() else 0
    return {"status": "ok", "model_loaded": loaded, "gpu_mb": mem_mb}
