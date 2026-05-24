import asyncio
import base64
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from llm_cuda import chat, stream_chat, get_model as get_llm
from stt import transcribe, get_model as get_stt
from tts_omnivoice import speak, speak_stream
from scores import ScoresStore
from vocabulary import get_qwen_vocab, whisper_to_qwen_id
from filter_stats import FilterStatsStore

app = FastAPI()


async def _speak_async(
    text: str, lang: str, speed: float, engine: str,
    slow: bool, instruct: str | None = None,
) -> bytes:
    return await asyncio.to_thread(speak, text, lang, speed, engine, slow, instruct)


@app.on_event("startup")
async def preload_models():
    await asyncio.to_thread(get_stt)
    await asyncio.to_thread(get_llm)
    ScoresStore.get()
    # OmniVoice loads lazily on first TTS call; loading all three at once exceeds 6 GB VRAM


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    reply: str
    latency_ms: float


class TranscribeResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/filter-stats")
def get_filter_stats():
    return {"tokens": FilterStatsStore.get().get_sorted()}


@app.post("/filter-stats/clear")
def clear_filter_stats():
    FilterStatsStore.get().clear()
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest):
    reply, elapsed = chat(req.message, req.system_prompt)
    return ChatResponse(reply=reply, latency_ms=round(elapsed * 1000, 1))


@app.post("/transcribe", response_model=TranscribeResponse)
async def post_transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    text, _ = transcribe(data)
    return TranscribeResponse(text=text)


@app.post("/speak")
async def post_speak(
    text: str = Query(...),
    lang: str = Query("zh"),
    engine: str = Query("omnivoice"),
):
    wav_bytes = await _speak_async(text, lang, 1.0, engine, False)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/speak-slow")
async def post_speak_slow(
    text: str = Query(...),
    lang: str = Query("zh"),
    engine: str = Query("omnivoice"),
    instruct: str = Query(None),
):
    wav_bytes = await _speak_async(text, lang, 0.65, engine, True, instruct)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/speak-stream")
async def post_speak_stream(
    text: str = Query(...),
    slow: bool = Query(False),
    instruct: str = Query(None),
):
    """Stream OmniVoice audio as float32 PCM.

    Response format: first 4 bytes = int32 LE sample rate, then float32 LE PCM samples.
    Note: OmniVoice generates the full clip first; this endpoint streams it as a single chunk.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)

    def _run_tts():
        try:
            for chunk in speak_stream(text, slow=slow, instruct=instruct or None):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    async def _stream_gen():
        loop.run_in_executor(None, _run_tts)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(_stream_gen(), media_type="application/octet-stream")


@app.get("/gpu-check")
def gpu_check():
    import torch
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 1),
    }


_TRANSLATE_SYSTEM = (
    "You are a translator. Return only the requested output, nothing else. "
    "No explanations, no extra text."
)

_TRANSLATE_PROMPTS = {
    "literal": (
        "Translate the following Chinese text word-by-word into English. "
        "Keep the exact Chinese word order. "
        "Return only the English glosses separated by spaces, one gloss per Chinese word.\n\n"
        "Chinese: {text}"
    ),
    "normal": (
        "Translate the following Chinese text naturally into English. "
        "Return only the translation.\n\n"
        "Chinese: {text}"
    ),
    "pinyin": (
        "Convert the following Chinese text to pinyin with tone marks. "
        "Connect syllables of the same word with a hyphen, separate words with spaces. "
        "Return only the pinyin.\n\n"
        "Chinese: {text}"
    ),
}


class TranslateRequest(BaseModel):
    text: str
    mode: Literal["literal", "normal", "pinyin"]


class TranslateResponse(BaseModel):
    text: str
    mode: str


def _do_translate(text: str, mode: str) -> str:
    prompt = _TRANSLATE_PROMPTS[mode].format(text=text)
    result, _ = chat(prompt, system_prompt=_TRANSLATE_SYSTEM)
    return result.strip()


@app.post("/translate", response_model=TranslateResponse)
async def post_translate(req: TranslateRequest):
    result = _do_translate(req.text, req.mode)
    return TranslateResponse(text=result, mode=req.mode)


def _update_scores(
    transcript: str,
    produced_llm_ids: list[int],
    pron_ranks_whisper: dict[int, int],
) -> None:
    _, tokenizer = get_llm()
    all_vocab = get_qwen_vocab()
    all_vocab_ids = set(all_vocab.keys())

    transcript_ids = set(tokenizer.encode(transcript)) & all_vocab_ids
    used_ids = transcript_ids | (set(produced_llm_ids) & all_vocab_ids)

    pron_ranks_qwen: dict[int, int] = {}
    for wid, rank in pron_ranks_whisper.items():
        qid = whisper_to_qwen_id(wid)
        if qid is not None and qid in all_vocab_ids:
            if qid not in pron_ranks_qwen or rank < pron_ranks_qwen[qid]:
                pron_ranks_qwen[qid] = rank

    ScoresStore.get().update_after_turn(all_vocab, used_ids, pron_ranks_qwen)


async def _stream_llm_to_ws(
    websocket: WebSocket,
    user_message: str,
    system_prompt: str | None,
    filter_vocab: bool = False,
    produced_token_ids: list[int] | None = None,
    history: list[dict] | None = None,
) -> str:
    full_response: list[str] = []
    for token_text in stream_chat(
        user_message, system_prompt,
        filter_vocab=filter_vocab,
        produced_token_ids=produced_token_ids,
        history=history,
    ):
        full_response.append(token_text)
        await websocket.send_json({"type": "token", "text": token_text})
    return "".join(full_response)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            system_prompt = msg.get("system_prompt")
            filter_vocab = bool(msg.get("filter_vocab", False))
            history: list[dict] | None = msg.get("history") or None

            if msg_type == "audio":
                audio_bytes = base64.b64decode(msg["data"])

                transcript, pron_ranks_whisper = await asyncio.to_thread(
                    transcribe, audio_bytes, filter_vocab
                )
                await websocket.send_json({"type": "transcript", "text": transcript})

                produced_llm_ids: list[int] = []
                response = await _stream_llm_to_ws(
                    websocket, transcript, system_prompt, filter_vocab, produced_llm_ids, history
                )
                await websocket.send_json({"type": "response_done", "text": response})

                update_task = asyncio.to_thread(
                    _update_scores, transcript, produced_llm_ids, pron_ranks_whisper
                )
                tts_task = _speak_async(response, "zh", 1.0, "omnivoice", False)
                _, wav_bytes = await asyncio.gather(update_task, tts_task)

                await websocket.send_json({
                    "type": "audio",
                    "data": base64.b64encode(wav_bytes).decode(),
                })

            elif msg_type == "text":
                user_message = msg["message"]
                await websocket.send_json({"type": "transcript", "text": user_message})

                produced_llm_ids: list[int] = []
                response = await _stream_llm_to_ws(
                    websocket, user_message, system_prompt, filter_vocab, produced_llm_ids, history
                )
                await websocket.send_json({"type": "response_done", "text": response})

                update_task = asyncio.to_thread(
                    _update_scores, user_message, produced_llm_ids, {}
                )
                tts_task = _speak_async(response, "zh", 1.0, "omnivoice", False)
                _, wav_bytes = await asyncio.gather(update_task, tts_task)

                await websocket.send_json({
                    "type": "audio",
                    "data": base64.b64encode(wav_bytes).decode(),
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
