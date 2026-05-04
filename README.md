# Language Tutor — Real-time Chinese Voice Conversation App

Real-time voice conversation app for learning Chinese (Mandarin / Traditional).  
All inference runs locally. No data leaves your machine.

---

## Current State (Steps 1–7 complete)

| Step | Component | Detail |
|------|-----------|--------|
| 1 | LLM | Qwen2.5-3B-Instruct-4bit on MLX via `generate_step` |
| 2 | STT | HuggingFace Whisper-small on MPS (Apple GPU) |
| 3 | TTS | Kokoro / Kokoro-MLX / Qwen3-TTS (selectable engine) |
| 4 | WebSocket | Single `/ws`, tokens stream live, per-step timing display |
| 5 | LLM logit filter | Vocabulary whitelist — unknown tokens masked to `-inf` |
| 6 | STT logit boost | `+3.0` log-space bias on known vocabulary tokens in Whisper |
| 7 | Dynamic vocab steering | Pronunciation scores drive soft biases; usage handicap prevents over-repetition |

**Typical latency (MacBook M1 16 GB):** STT 1.3 s · TTFT 607 ms · LLM 1.3 s · TTS 1.5 s · Total ~4 s

---

## Architecture

```
Browser (Vite TypeScript)
  ↕ WebSocket
FastAPI Python backend
  ├── faster-whisper    STT + logit boost
  ├── mlx-lm Qwen2.5   LLM + logit filter + vocab steering
  ├── Kokoro / Qwen3    TTS (Apple Silicon)
  └── scores.py         pronunciation scores → bias recalculation
```

---

## Platform Support

| Platform | Status |
|----------|--------|
| macOS Apple Silicon (M1/M2/M3) | ✅ Fully working |
| Ubuntu RTX 2060 (CUDA) | ✅ Working — `main_ubuntu.py` |

**Ubuntu stack:**
- LLM: `Qwen/Qwen2.5-3B-Instruct` via `transformers` + `bitsandbytes` 4-bit CUDA — logit filter + dynamic bias preserved via custom `LogitsProcessor`
- TTS: **OmniVoice** (`k2-fsa/OmniVoice`, diffusion, ~0.1 RTF @ 16 steps) in `design` mode with Mandarin instruct
- STT: HuggingFace Whisper on CUDA — logit capturer + vocab locker unchanged

---

## Setup (Ubuntu RTX 2060 / CUDA)

```bash
# Reuse the omnivoice-tts venv (already has torch, transformers, omnivoice, etc.)
source /home/jani/dev/typescript/omnivoice-tts/server/.venv/bin/activate

# Install the two missing packages
pip install bitsandbytes anthropic

# Frontend
cd frontend && npm install

# Run backend
cd backend
uvicorn main_ubuntu:app --host 0.0.0.0 --port 8000

# Run frontend (separate terminal)
cd frontend && npm run dev
```

GPU memory budget (RTX 2060 6 GB):
- Whisper-small: ~0.1 GB
- Qwen2.5-3B 4-bit: ~1.5–2 GB
- OmniVoice: ~2–3 GB  ← loaded on demand; `empty_cache()` called after each generate

---

## Setup (macOS M1)

```bash
# Python
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install mlx mlx-lm mlx-audio faster-whisper kokoro soundfile fastapi uvicorn websockets anthropic

# Frontend
cd frontend
npm install

# Run backend
cd backend
source ../venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Run frontend (separate terminal)
cd frontend
npm run dev
```

---

## Dynamic Vocabulary Steering (Step 7)

Three soft signals update after every turn:

| Signal | Effect |
|--------|--------|
| New word introduced | `+5.0` initial boost |
| Word not used this turn | `+0.1` nudge |
| Word used this turn | `-0.1` handicap |
| Poor pronunciation (Whisper logit gap) | `+0.1` to `+0.5` boost |

All boosts are capped so the model is *guided* toward new/hard words without being forced — conversation still flows naturally.

---

## Step 8 (Next)

Every 10 exchanges → Claude API → updated system prompt injected into Qwen context.  
Adapts conversation topics and vocabulary to the learner's demonstrated level.
