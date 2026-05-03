import './style.css'
import { startRecording, stopRecording, playAudioBase64, playAudioStream } from './audio'

const WS_URL = 'ws://localhost:8000/ws'
const API_URL = 'http://localhost:8000'

// ── Settings ──────────────────────────────────────────────────────────────────
type TtsEngine = 'kokoro' | 'kokoro-mlx' | 'qwen3-tts'

let saveAudio    = localStorage.getItem('lt_save_audio') === 'true'
let filterVocab  = localStorage.getItem('lt_vocab_filter') === 'true'
let ttsEnginePlay = (localStorage.getItem('lt_tts_engine_play') ?? 'kokoro') as TtsEngine
let ttsEngineSlow = (localStorage.getItem('lt_tts_engine_slow') ?? 'kokoro') as TtsEngine
let autoGenSlow   = localStorage.getItem('lt_auto_gen_slow') === 'true'
let slowInstruct  = localStorage.getItem('lt_slow_instruct') ?? '说话慢一点，发音清晰，语气耐心'

function persistSettings() {
  localStorage.setItem('lt_save_audio',       String(saveAudio))
  localStorage.setItem('lt_vocab_filter',     String(filterVocab))
  localStorage.setItem('lt_tts_engine_play',  ttsEnginePlay)
  localStorage.setItem('lt_tts_engine_slow',  ttsEngineSlow)
  localStorage.setItem('lt_auto_gen_slow',    String(autoGenSlow))
  localStorage.setItem('lt_slow_instruct',    slowInstruct)
}

// ── HTML ──────────────────────────────────────────────────────────────────────
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="chat-container">
    <div class="header-row">
      <h1>Language Tutor</h1>
      <button id="settings-btn" class="icon-btn" title="Settings">⚙</button>
    </div>

    <div id="settings-overlay" class="settings-overlay hidden">
      <div id="settings-panel" class="settings-panel">
        <h2>Settings</h2>
        <label class="setting-row">
          <input type="checkbox" id="vocab-filter-cb" ${filterVocab ? 'checked' : ''} />
          Vocab filter
        </label>
        <label class="setting-row">
          <input type="checkbox" id="save-audio-cb" ${saveAudio ? 'checked' : ''} />
          Save audio
        </label>
        <div class="setting-row setting-select-row">
          <span class="setting-label">▶ Play engine</span>
          <select id="tts-engine-play-sel" class="setting-select">
            <option value="kokoro"     ${ttsEnginePlay === 'kokoro'     ? 'selected' : ''}>Kokoro</option>
            <option value="kokoro-mlx" ${ttsEnginePlay === 'kokoro-mlx' ? 'selected' : ''}>Kokoro MLX ⚡</option>
            <option value="qwen3-tts"  ${ttsEnginePlay === 'qwen3-tts'  ? 'selected' : ''}>Qwen3 TTS ⚡</option>
          </select>
        </div>
        <div class="setting-row setting-select-row">
          <span class="setting-label">🐢 Slow engine</span>
          <select id="tts-engine-slow-sel" class="setting-select">
            <option value="kokoro"     ${ttsEngineSlow === 'kokoro'     ? 'selected' : ''}>Kokoro</option>
            <option value="kokoro-mlx" ${ttsEngineSlow === 'kokoro-mlx' ? 'selected' : ''}>Kokoro MLX ⚡</option>
            <option value="qwen3-tts"  ${ttsEngineSlow === 'qwen3-tts'  ? 'selected' : ''}>Qwen3 TTS ⚡ (teacher)</option>
          </select>
        </div>
        <div class="setting-col">
          <span class="setting-label">🐢 Slow style instruction (emotion/pace)</span>
          <textarea id="slow-instruct-ta" class="setting-textarea" rows="2" placeholder="e.g. 说话慢一点，发音清晰">${slowInstruct}</textarea>
        </div>
        <label class="setting-row">
          <input type="checkbox" id="auto-gen-slow-cb" ${autoGenSlow ? 'checked' : ''} />
          Auto-generate 🐢 slow audio after ▶ plays
        </label>
      </div>
    </div>

    <div id="messages" class="messages"></div>

    <div class="input-row">
      <input id="msg-input" type="text" placeholder="Type or use mic…" />
      <button id="send-btn">Send</button>
      <button id="mic-btn" class="mic-btn" title="Click to record">🎤</button>
    </div>
    <div id="status" class="status">Connecting…</div>
    <div id="timings" class="timings"></div>
  </div>
`

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl = document.getElementById('messages')!
const input = document.getElementById('msg-input') as HTMLInputElement
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement
const micBtn = document.getElementById('mic-btn') as HTMLButtonElement
const statusEl = document.getElementById('status')!
const timingsEl = document.getElementById('timings')!
const settingsBtn = document.getElementById('settings-btn') as HTMLButtonElement
const settingsOverlay = document.getElementById('settings-overlay') as HTMLDivElement
const settingsPanel = document.getElementById('settings-panel') as HTMLDivElement
const vocabFilterCb      = document.getElementById('vocab-filter-cb')       as HTMLInputElement
const saveAudioCb        = document.getElementById('save-audio-cb')         as HTMLInputElement
const ttsEnginePlaySel   = document.getElementById('tts-engine-play-sel')   as HTMLSelectElement
const ttsEngineSlowSel   = document.getElementById('tts-engine-slow-sel')   as HTMLSelectElement
const autoGenSlowCb      = document.getElementById('auto-gen-slow-cb')      as HTMLInputElement
const slowInstructTa     = document.getElementById('slow-instruct-ta')      as HTMLTextAreaElement

// ── Settings wiring ───────────────────────────────────────────────────────────
settingsBtn.addEventListener('click', (e) => {
  e.stopPropagation()
  settingsOverlay.classList.toggle('hidden')
})

settingsOverlay.addEventListener('click', (e) => {
  if (!settingsPanel.contains(e.target as Node)) {
    settingsOverlay.classList.add('hidden')
  }
})

vocabFilterCb.addEventListener('change', () => {
  filterVocab = vocabFilterCb.checked
  persistSettings()
})

saveAudioCb.addEventListener('change', () => {
  saveAudio = saveAudioCb.checked
  persistSettings()
})

ttsEnginePlaySel.addEventListener('change', () => {
  ttsEnginePlay = ttsEnginePlaySel.value as TtsEngine
  persistSettings()
})

ttsEngineSlowSel.addEventListener('change', () => {
  ttsEngineSlow = ttsEngineSlowSel.value as TtsEngine
  persistSettings()
})

autoGenSlowCb.addEventListener('change', () => {
  autoGenSlow = autoGenSlowCb.checked
  persistSettings()
})

slowInstructTa.addEventListener('input', () => {
  slowInstruct = slowInstructTa.value
  persistSettings()
})

// ── Helpers ───────────────────────────────────────────────────────────────────
function s(ms: number) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}
function setStatus(msg: string) { statusEl.textContent = msg }
function setTimings(parts: { label: string; ms: number }[]) {
  timingsEl.textContent = parts.map((p) => `${p.label}: ${s(p.ms)}`).join('  |  ')
}
function clearTimings() { timingsEl.textContent = '' }
function setInputsDisabled(disabled: boolean) {
  sendBtn.disabled = disabled
  input.disabled = disabled
  micBtn.disabled = disabled
}

let userScrolledUp = false

messagesEl.addEventListener('scroll', () => {
  const distFromBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight
  userScrolledUp = distFromBottom > 40
})

function scrollToBottomIfNear() {
  if (!userScrolledUp) messagesEl.scrollTop = messagesEl.scrollHeight
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin)
}

function trySaveAudioToStorage(key: string, b64: string) {
  if (!saveAudio) return
  try {
    localStorage.setItem(key, b64)
  } catch {
    // quota exceeded — silently skip
  }
}

// ── Bubble factory ────────────────────────────────────────────────────────────
type BubbleHandle = ReturnType<typeof createBubble>

function createBubble(role: 'user' | 'assistant', initialText = '') {
  let bubbleText = initialText
  let normalAudioB64: string | null = null
  let slowAudioB64: string | null = null
  const cachedExpansions = new Map<string, string>()
  const activeExpansions = new Set<string>()

  // Outer wrapper
  const el = document.createElement('div')
  el.className = `message ${role}`

  const textEl = document.createElement('div')
  textEl.className = 'bubble-text'
  textEl.textContent = bubbleText

  const actionsEl = document.createElement('div')
  actionsEl.className = 'bubble-actions'

  const expansionsEl = document.createElement('div')
  expansionsEl.className = 'bubble-expansions'

  el.append(textEl, actionsEl, expansionsEl)
  messagesEl.appendChild(el)
  userScrolledUp = false
  messagesEl.scrollTop = messagesEl.scrollHeight

  // ── button helpers ──
  function makeBtn(label: string, title: string, extraClass = '') {
    const b = document.createElement('button')
    b.className = `bubble-btn${extraClass ? ' ' + extraClass : ''}`
    b.textContent = label
    b.title = title
    return b
  }

  // ── Fetch slow audio in background (used by auto-gen) ──
  async function fetchSlowAudio() {
    if (slowAudioB64 || !bubbleText) return
    const params = new URLSearchParams({
      text: bubbleText, lang: 'zh', engine: ttsEngineSlow, instruct: slowInstruct
    })
    const res = await fetch(`${API_URL}/speak-slow?${params}`, { method: 'POST' })
    const b64 = arrayBufferToBase64(await res.arrayBuffer())
    slowAudioB64 = b64
    trySaveAudioToStorage(`lt_audio_slow_${bubbleText.slice(0, 40)}`, b64)
  }

  // ── Play (normal) ──
  const playBtn = makeBtn('▶', 'Play')
  playBtn.addEventListener('click', async () => {
    if (!bubbleText) return
    if (normalAudioB64) { await playAudioBase64(normalAudioB64); return }
    playBtn.disabled = true
    try {
      let b64: string
      if (ttsEnginePlay === 'qwen3-tts') {
        const params = new URLSearchParams({ text: bubbleText })
        b64 = await playAudioStream(`${API_URL}/speak-stream?${params}`)
      } else {
        const res = await fetch(
          `${API_URL}/speak?text=${encodeURIComponent(bubbleText)}&lang=zh&engine=${ttsEnginePlay}`,
          { method: 'POST' }
        )
        b64 = arrayBufferToBase64(await res.arrayBuffer())
        await playAudioBase64(b64)
      }
      normalAudioB64 = b64
      trySaveAudioToStorage(`lt_audio_${bubbleText.slice(0, 40)}`, b64)
      if (autoGenSlow) fetchSlowAudio()
    } finally {
      playBtn.disabled = false
    }
  })

  // ── Slow play ──
  const slowBtn = makeBtn('🐢', 'Play slowly')
  slowBtn.addEventListener('click', async () => {
    if (!bubbleText) return
    if (slowAudioB64) { await playAudioBase64(slowAudioB64); return }
    slowBtn.disabled = true
    try {
      if (ttsEngineSlow === 'qwen3-tts') {
        const params = new URLSearchParams({ text: bubbleText, slow: 'true', instruct: slowInstruct })
        slowAudioB64 = await playAudioStream(`${API_URL}/speak-stream?${params}`)
        trySaveAudioToStorage(`lt_audio_slow_${bubbleText.slice(0, 40)}`, slowAudioB64)
      } else {
        await fetchSlowAudio()
        await playAudioBase64(slowAudioB64!)
      }
    } finally {
      slowBtn.disabled = false
    }
  })

  // ── Expansion buttons (literal / normal / pinyin) ──
  function makeExpansionBtn(label: string, title: string, mode: string) {
    const b = makeBtn(label, title)
    b.addEventListener('click', async () => {
      // Toggle off
      if (activeExpansions.has(mode)) {
        activeExpansions.delete(mode)
        expansionsEl.querySelector(`[data-mode="${mode}"]`)?.remove()
        b.classList.remove('active')
        return
      }
      b.disabled = true
      try {
        let result = cachedExpansions.get(mode)
        if (!result) {
          const res = await fetch(`${API_URL}/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: bubbleText, mode }),
          })
          const data = await res.json()
          result = (data.text as string) || '—'
          cachedExpansions.set(mode, result)
        }
        activeExpansions.add(mode)
        b.classList.add('active')
        const expEl = document.createElement('div')
        expEl.className = `expansion expansion-${mode}`
        expEl.dataset.mode = mode
        expEl.textContent = result
        expansionsEl.appendChild(expEl)
        scrollToBottomIfNear()
      } finally {
        b.disabled = false
      }
    })
    return b
  }

  actionsEl.append(
    playBtn,
    slowBtn,
    makeExpansionBtn('字', 'Word-by-word translation', 'literal'),
    makeExpansionBtn('译', 'Natural translation', 'normal'),
    makeExpansionBtn('拼', 'Pinyin', 'pinyin'),
  )

  return {
    el,
    textEl,
    appendToken(token: string) {
      bubbleText += token
      textEl.textContent = bubbleText
      scrollToBottomIfNear()
    },
    setText(t: string) {
      bubbleText = t
      textEl.textContent = t
    },
    // Called when we already have TTS audio (AI response path)
    setNormalAudio(b64: string) {
      normalAudioB64 = b64
      trySaveAudioToStorage(`lt_audio_${bubbleText.slice(0, 40)}`, b64)
    },
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws: WebSocket | null = null
let isRecording = false
let streamingBubble: BubbleHandle | null = null
let pendingAiBubble: BubbleHandle | null = null  // bubble waiting for audio attachment

let tSend = 0, tTranscript = 0, tFirstToken = 0, tResponseDone = 0

function connectWebSocket() {
  ws = new WebSocket(WS_URL)

  ws.onopen = () => {
    setStatus('Ready')
    setInputsDisabled(false)
  }

  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data as string)

    switch (msg.type) {
      case 'transcript': {
        tTranscript = performance.now()
        createBubble('user', msg.text)
        setStatus('Thinking…')
        setTimings([{ label: 'STT', ms: Math.round(tTranscript - tSend) }])
        streamingBubble = null
        pendingAiBubble = null
        tFirstToken = 0
        break
      }
      case 'token': {
        if (!streamingBubble) {
          tFirstToken = performance.now()
          streamingBubble = createBubble('assistant', '')
          pendingAiBubble = streamingBubble
          setStatus('Generating…')
          setTimings([
            { label: 'STT', ms: Math.round(tTranscript - tSend) },
            { label: 'TTFT', ms: Math.round(tFirstToken - tTranscript) },
          ])
        }
        streamingBubble.appendToken(msg.text)
        break
      }
      case 'response_done': {
        tResponseDone = performance.now()
        streamingBubble = null
        setStatus('Synthesising…')
        setTimings([
          { label: 'STT', ms: Math.round(tTranscript - tSend) },
          { label: 'TTFT', ms: Math.round(tFirstToken - tTranscript) },
          { label: 'LLM', ms: Math.round(tResponseDone - tTranscript) },
        ])
        break
      }
      case 'audio': {
        const tAudioReady = performance.now()
        const ttsMs = Math.round(tAudioReady - tResponseDone)
        const totalMs = Math.round(tAudioReady - tSend)
        setStatus('Playing…')
        setTimings([
          { label: 'STT', ms: Math.round(tTranscript - tSend) },
          { label: 'TTFT', ms: Math.round(tFirstToken - tTranscript) },
          { label: 'LLM', ms: Math.round(tResponseDone - tTranscript) },
          { label: 'TTS', ms: ttsMs },
          { label: 'Total', ms: totalMs },
        ])
        // Attach audio to the AI bubble so its ▶ button works immediately
        pendingAiBubble?.setNormalAudio(msg.data)
        pendingAiBubble = null
        await playAudioBase64(msg.data)
        setStatus('Ready')
        setInputsDisabled(false)
        input.focus()
        break
      }
      case 'error': {
        setStatus(`Error: ${msg.message}`)
        setInputsDisabled(false)
        break
      }
    }
  }

  ws.onclose = () => {
    setStatus('Disconnected — reconnecting…')
    setInputsDisabled(true)
    setTimeout(connectWebSocket, 2000)
  }

  ws.onerror = () => {
    setStatus('Connection error')
  }
}

// ── Text send ─────────────────────────────────────────────────────────────────
async function handleTextSend() {
  const text = input.value.trim()
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return
  input.value = ''
  setInputsDisabled(true)
  clearTimings()
  tSend = performance.now()
  ws.send(JSON.stringify({ type: 'text', message: text, filter_vocab: filterVocab, tts_engine: ttsEnginePlay }))
}

sendBtn.addEventListener('click', handleTextSend)
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleTextSend() })

// ── Mic ───────────────────────────────────────────────────────────────────────
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve((reader.result as string).split(',')[1])
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

micBtn.addEventListener('click', async () => {
  if (isRecording) {
    isRecording = false
    micBtn.classList.remove('recording')
    micBtn.textContent = '🎤'
    setStatus('Processing…')
    setInputsDisabled(true)
    clearTimings()
    try {
      const blob = await stopRecording()
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        setStatus('Not connected')
        setInputsDisabled(false)
        return
      }
      tSend = performance.now()
      const b64 = await blobToBase64(blob)
      ws.send(JSON.stringify({ type: 'audio', data: b64, filter_vocab: filterVocab, tts_engine: ttsEnginePlay }))
    } catch (e) {
      setStatus(`Error: ${e}`)
      setInputsDisabled(false)
    }
  } else {
    isRecording = true
    micBtn.classList.add('recording')
    micBtn.textContent = '⏹'
    setStatus('Recording… click ⏹ to stop')
    try {
      await startRecording()
    } catch (e) {
      isRecording = false
      micBtn.classList.remove('recording')
      micBtn.textContent = '🎤'
      setStatus(`Mic error: ${e}`)
    }
  }
})

connectWebSocket()
