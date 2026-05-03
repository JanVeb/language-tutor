```
You are implementing a language learning tutor app step by step.
Test each step before proceeding to next.
Never proceed to next step without explicit confirmation from me.

## Setup
- Create project folder: language-tutor
- Install pyenv if not present: brew install pyenv
- Install Python 3.11: pyenv install 3.11.9
- Set local Python: pyenv local 3.11.9
- Create venv: python -m venv venv
- Activate: source venv/bin/activate
- All packages install inside venv only
- Create frontend: npm create vite@latest frontend -- --template vanilla-ts

## Stack
- Frontend: Vite vanilla TypeScript
- Backend: Python FastAPI
- Target hardware: MacBook M1 16GB RAM unified memory

## Goal
Real-time voice conversation app for learning Chinese.
All inference runs locally using Apple Silicon MLX framework.
MLX exploits M1 unified memory - CPU and GPU share same RAM,
no data copying between them. Fastest possible local inference on Mac.
Cloud LLM called periodically to improve conversation quality.

## Why MLX over llama.cpp
- Built by Apple specifically for M-series chips
- Unified memory means zero copy between CPU and GPU
- 20-40% faster than llama.cpp on same hardware
- Same models work, just faster

## Models
- LLM: Qwen2.5 3B 4-bit quantized
  - Why: Best Chinese/English model at this size
  - Source: mlx-community/Qwen2.5-3B-Instruct-4bit on HuggingFace
  - Framework: mlx-lm (Apple's own LLM library)
- STT: faster-whisper medium model
  - Why: Best Python logit access, needed for pronunciation scoring
  - C++ backend via ctranslate2, fast on M1
- TTS: Kokoro
  - Why: Lightest fastest TTS available, good quality

## Core libraries
- mlx-lm: LLM inference + logit access on Apple Silicon
- faster-whisper: speech recognition + logit access
- FastAPI + websockets: backend API
- anthropic: Claude API for periodic meta-learning

## Architecture
Frontend (Vite TypeScript)
  ↕ WebSocket
Backend (FastAPI Python)
  ├── faster-whisper     (STT + logit access)
  ├── mlx-lm Qwen2.5 3B (LLM + logit access)
  └── Kokoro             (TTS)

## File structure
language-tutor/
  backend/
    main.py          - FastAPI app + WebSocket server
    llm.py           - mlx-lm wrapper + logit filtering
    stt.py           - faster-whisper wrapper + logit bias
    tts.py           - Kokoro wrapper
    vocabulary.py    - known words tracker + token ids
    meta_learner.py  - Claude API periodic updater
  frontend/
    src/
      main.ts        - UI and WebSocket client
      audio.ts       - microphone recording + audio playback
      vocabulary.ts  - user vocabulary state

## Build order - STOP at each step, wait for my confirmation

### Step 1 - LLM chat working
- Install mlx and mlx-lm:
  pip install mlx mlx-lm
- Download model:
  mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct --quantize --q-bits 4
  or download directly: mlx_lm.generate --model mlx-community/Qwen2.5-3B-Instruct-4bit
- FastAPI POST /chat: receives text, returns text response
- Simple TypeScript fetch call to test in browser
STOP: measure response time, target under 1 second

### Step 2 - Add STT
- Install faster-whisper:
  pip install faster-whisper
- FastAPI POST /transcribe: receives audio blob, returns text
- TypeScript: record microphone via MediaRecorder, send blob, display transcription
STOP: test accuracy in English first, then Chinese

### Step 3 - Add TTS
- Install Kokoro:
  pip install kokoro
- FastAPI POST /speak: receives text, returns audio blob
- TypeScript: receive audio, play it back automatically
STOP: full voice loop working - speak → transcribe → LLM → speak back

### Step 4 - WebSocket streaming
- Replace HTTP endpoints with single WebSocket connection
- Stream LLM tokens to frontend as they generate
- Full pipeline over WebSocket:
  record → send audio → transcribe → LLM stream → TTS → play
STOP: measure full round trip latency, should feel natural

### Step 5 - LLM logit filtering
- In vocabulary.py: maintain known_vocabulary as list of token ids
- In llm.py: intercept raw logits on each forward pass via mlx-lm
- Zero out logit scores for token ids not in known_vocabulary
- Only known tokens can be sampled
STOP: test with small hardcoded vocabulary list, verify output only uses known words

### Step 6 - STT logit bias
- In stt.py: pass known vocabulary token ids to faster-whisper
- Boost probability of known vocabulary tokens during transcription
- Helps Whisper recognize words learner is attempting to say
- Especially useful when learner mispronounces slightly
STOP: test by saying known words with exaggerated bad pronunciation

### Step 7 - Pronunciation scorer
- Before vocabulary boost: capture raw Whisper logits per word
- After vocabulary boost: capture adjusted logits per word
- Difference = how hard Whisper had to work to recognize the word
- Large difference = poor pronunciation
- Small difference = good pronunciation
- Store score per word in scores.json, update on each attempt
STOP: speak same word clearly then poorly, verify scores differ correctly

### Step 7.1 Add function buttons
- every chat bubble should have 2 play buttons, translate literal, translate normal buttons, pinyin buttons under it, both user and ai chat bubles
- second play button should have play with turtle icon, when presed tts should be instructed to generate new version of same string llm returned in bubble but with instructions to pronaunce it in parenting style voice, slowly, expresivly
- if play is presed, it plays audio (need to save audios localy and pas their name/id to coresponding chat window play button).
- if literal translation button is pressed, llm model should be instructed to return same sentance, word by word translated into user language, with template, 
user: 歡迎你來台灣
llm: 歡迎你來台灣
just that when returning it should be instructed to limit response to user language so it is literary translated, or how to do this?
- when normal translate button is presed, model should translate sentance how peaople translate
- pinyin button should transalte it to pinyin, we can do that with qwen already, it has pinyin tokens
user: 歡迎你來台灣
LLM： huānyíng nǐ lái táiwān
ideal would be perfect if llm could return pinyin words conected so its easy to see words from pinyin structure, can llm do that?
- all translations and pinyin should be atached to same chat bubble, so it expands, buttons always showing under the bubble.
- add settings separate window, in it add check box named sasve audio,, uts state should be saved in local memory for persistence, also vocab filter check box should be moved to settings window and also have persistent local memory
- settings button should be placed at the top of the app, it should open close settings window, also clicking anywhere outside settings window, should close settings window as well


### Step 8 - Cloud meta-learning loop
- Every 10 conversation exchanges:
  → send conversation history to Claude API
  → Claude analyzes: vocabulary gaps, user interests, engagement level
  → Claude returns updated system prompt guiding local LLM
  → inject new system prompt into Qwen context
- This makes local model smarter without running large model locally
STOP: have 10 exchange conversation, verify system prompt updates, 
      verify next conversation feels more personalized

## At every STOP
- Report response latency at each stage
- Report whether interaction feels natural
- Show any errors in full
- Wait for my explicit go ahead before next step

## Key principle
Each step must work correctly and feel good before adding complexity.
A working simple version beats a broken complex version every time.
```

---

One heads up — MLX logit access API is slightly less documented than llama.cpp. If the agent struggles at Step 5, tell it to use `mlx_lm.utils.generate_step` which gives per-token logit access. That's the right function.



***
What's built (Steps 1–6 complete)
Step	Component	Detail
1	LLM	Qwen2.5-3B-Instruct-4bit on MLX via generate_step
2	STT	HuggingFace Whisper-small on MPS (Apple GPU) — 1.3s
3	TTS	Kokoro, Chinese voice zf_xiaobei
4	WebSocket	Single /ws, tokens stream live, per-step timing display
5	LLM logit filter	-inf mask on non-vocab tokens, toggle in UI
6	STT logit boost	+3.0 log-space bias on known Whisper tokens
Timing: STT 1.3s | TTFT 607ms | LLM 1.3s | TTS 1.5s | Total 4.0s

Step 7 — Dynamic vocabulary steering (your enhanced design)
Three interlocking signals per token:


final_bias[token] = + pronunciation_boost   ← poor score → say it more
                  - usage_handicap          ← used often → say it less
Pronunciation boost: captured from raw Whisper logits before the Step 6 boost applies — low raw confidence = Whisper struggled = poor pronunciation = boost goes up
Usage handicap: grows each time the token appears in any turn, capped at MAX_HANDICAP so the word stays accessible
After every exchange: parse both the user transcript and LLM response, update counts and recalculate the bias array for next turn
No hard suppression — all vocab words remain possible, just steered by soft floating biases
This replaces spaced repetition — the model naturally gravitates toward words that are new, rare, or pronounced badly

after every user or llm response, tokens that were used get slight hendikep -0.1, and tokens that werent used get slight positive nudge of 0.1
while new words initialy get nudge +5, and mispronounced words etc based on logit score from step 7, get +0.1 to +0.5 nudge based on how far down they were from the first sugested word/token

both new words and words that werent used much or words that got bad score should have max cap of boost, we dont want model to place 5 new words in one sentance, so user can undersand what was said, positive boost should just drive llm to use new words, low score words more offten while not forcing it to over use them so conversation flows easill for user

New file needed: backend/scores.py — persistence to scores.json, bias recalculation, the post-turn update hook.

Step 8 (unchanged)
Every 10 exchanges → Claude API → updated system prompt injected into Qwen context.