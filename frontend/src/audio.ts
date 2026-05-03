let mediaRecorder: MediaRecorder | null = null
let audioChunks: Blob[] = []

export async function startRecording(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  audioChunks = []
  mediaRecorder = new MediaRecorder(stream)
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) audioChunks.push(e.data)
  }
  mediaRecorder.start()
}

export function stopRecording(): Promise<Blob> {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder) return reject('Not recording')
    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' })
      mediaRecorder?.stream.getTracks().forEach((t) => t.stop())
      mediaRecorder = null
      resolve(blob)
    }
    mediaRecorder.stop()
  })
}

/** Stream float32 PCM from `/speak-stream`, play immediately via Web Audio API.
 *  Returns base64-encoded WAV of the full response for caching. */
export async function playAudioStream(url: string, method = 'POST'): Promise<string> {
  const response = await fetch(url, { method })
  if (!response.ok || !response.body) throw new Error(`TTS stream error: ${response.status}`)

  const reader = response.body.getReader()
  const ctx = new AudioContext()
  if (ctx.state === 'suspended') await ctx.resume()

  let sampleRate = 24000
  let headerRead = false
  let pending: Uint8Array<ArrayBufferLike> = new Uint8Array(0)
  let nextTime = ctx.currentTime + 0.08
  const allChunks: Float32Array[] = []

  function merge(a: Uint8Array<ArrayBufferLike>, b: Uint8Array<ArrayBufferLike>): Uint8Array {
    const out = new Uint8Array(a.length + b.length)
    out.set(a); out.set(b, a.length)
    return out
  }

  function scheduleAndCollect(data: Uint8Array): void {
    const n = Math.floor(data.length / 4)
    if (n === 0) return
    const floats = new Float32Array(data.buffer as ArrayBuffer, data.byteOffset, n)
    allChunks.push(floats.slice())
    const buf = ctx.createBuffer(1, n, sampleRate)
    buf.copyToChannel(floats, 0)
    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)
    src.start(nextTime)
    nextTime += buf.duration
  }

  while (true) {
    const { done, value } = await reader.read()
    if (value) {
      pending = merge(pending, value)
      if (!headerRead && pending.length >= 4) {
        sampleRate = new DataView(pending.buffer, pending.byteOffset, 4).getInt32(0, true)
        pending = pending.slice(4)
        headerRead = true
      }
      if (headerRead) {
        const chunkBytes = 8192 * 4
        while (pending.length >= chunkBytes) {
          scheduleAndCollect(pending.slice(0, chunkBytes))
          pending = pending.slice(chunkBytes)
        }
      }
    }
    if (done) break
  }
  if (headerRead && pending.length >= 4) scheduleAndCollect(pending)

  const remaining = nextTime - ctx.currentTime
  if (remaining > 0) await new Promise(r => setTimeout(r, remaining * 1000 + 150))
  await ctx.close()

  // Build WAV from accumulated float32 chunks for caching / replay
  const totalSamples = allChunks.reduce((s, c) => s + c.length, 0)
  const combined = new Float32Array(totalSamples)
  let off = 0
  for (const c of allChunks) { combined.set(c, off); off += c.length }
  return _float32ToWavBase64(combined, sampleRate)
}

function _float32ToWavBase64(samples: Float32Array, sampleRate: number): string {
  const int16 = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)))
  }
  const header = new ArrayBuffer(44)
  const v = new DataView(header)
  const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  ws(0, 'RIFF'); v.setUint32(4, 36 + int16.byteLength, true); ws(8, 'WAVE')
  ws(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true)
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true)
  v.setUint16(32, 2, true); v.setUint16(34, 16, true)
  ws(36, 'data'); v.setUint32(40, int16.byteLength, true)
  const wav = new Uint8Array(44 + int16.byteLength)
  wav.set(new Uint8Array(header))
  wav.set(new Uint8Array(int16.buffer as ArrayBuffer), 44)
  let bin = ''; for (const b of wav) bin += String.fromCharCode(b)
  return btoa(bin)
}

export async function playAudioBase64(base64Data: string): Promise<void> {
  const bytes = atob(base64Data)
  const arr = new Uint8Array(bytes.length)
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
  const blob = new Blob([arr], { type: 'audio/wav' })
  const url = URL.createObjectURL(blob)
  const audio = new Audio(url)
  return new Promise((resolve) => {
    audio.onended = () => { URL.revokeObjectURL(url); resolve() }
    audio.onerror = () => { URL.revokeObjectURL(url); resolve() }
    audio.play()
  })
}
