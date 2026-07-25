import { useState, useRef, useEffect, FormEvent, ChangeEvent } from 'react'

// ── Types ──────────────────────────────────────────────────

type Tab = 'generate' | 'train' | 'history'

type LoraInfo = { name: string; path: string; size_bytes: number }

type Job = {
  id: string
  prompt: string
  status: string
  resultUrl?: string
  error?: string
  createdAt: string
}

// ── Helpers ────────────────────────────────────────────────

const INFERENCE_API = '/api/v1'
const WEB_API = '/api/generation'

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      resolve(result.split(',')[1]) // strip data:image/...;base64,
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

// ── App ─────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<Tab>('generate')
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* Header */}
        <header className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">mflux</h1>
            <p className="text-gray-400 text-sm mt-0.5">Image generation on Apple Silicon</p>
          </div>
        </header>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 border-b border-gray-800">
          {(['generate', 'train', 'history'] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
                tab === t
                  ? 'bg-gray-800 text-white border-b-2 border-blue-500'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {t === 'generate' ? '🎨 Generate' : t === 'train' ? '🧠 Train LoRA' : '📋 History'}
            </button>
          ))}
        </div>

        {tab === 'generate' && <GenerateTab />}
        {tab === 'train' && <TrainTab />}
        {tab === 'history' && <HistoryTab />}
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  Generate Tab
// ═══════════════════════════════════════════════════════════════

function GenerateTab() {
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [steps, setSteps] = useState(25)
  const [guidance, setGuidance] = useState(3.5)
  const [loras, setLoras] = useState<LoraInfo[]>([])
  const [selectedLoras, setSelectedLoras] = useState<string[]>([])
  const [loraScales, setLoraScales] = useState<Record<string, number>>({})
  const [refImage, setRefImage] = useState<File | null>(null)
  const [refPreview, setRefPreview] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [resultId, setResultId] = useState<string | null>(null)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [resultError, setResultError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Fetch available LoRAs on mount
  useEffect(() => {
    fetch(`${INFERENCE_API}/lora/list`)
      .then(r => r.json())
      .then(setLoras)
      .catch(() => {})
  }, [])

  // Cleanup polling on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const toggleLora = (path: string) => {
    setSelectedLoras(prev =>
      prev.includes(path) ? prev.filter(p => p !== path) : [...prev, path]
    )
    if (!loraScales[path]) {
      setLoraScales(s => ({ ...s, [path]: 0.8 }))
    }
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!prompt.trim()) return
    setSubmitting(true)
    setResultImage(null)
    setResultError(null)
    setResultId(null)

    try {
      const body: Record<string, unknown> = {
        prompt: prompt.trim(),
        num_inference_steps: steps,
        guidance_scale: guidance,
      }
      if (negativePrompt.trim()) body.negative_prompt = negativePrompt.trim()
      if (selectedLoras.length > 0) {
        body.lora_paths = selectedLoras
        body.lora_scales = selectedLoras.map(p => loraScales[p] ?? 0.8)
      }
      if (refImage) {
        body.reference_image_b64 = await readFileAsBase64(refImage)
        body.denoise_strength = 0.4
      }

      const res = await fetch(`${INFERENCE_API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      setResultId(data.job_id)
      startPolling(data.job_id)
    } catch (err) {
      setResultError('Failed to submit generation')
    } finally {
      setSubmitting(false)
    }
  }

  const startPolling = (id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${INFERENCE_API}/jobs/${id}`)
        const data = await res.json()
        if (data.status === 'completed') {
          setResultImage(`${INFERENCE_API}/jobs/${id}/image?t=${Date.now()}`)
          if (pollRef.current) clearInterval(pollRef.current)
        } else if (data.status === 'failed') {
          setResultError(data.error || 'Generation failed')
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current)
      }
    }, 2000)
  }

  const handleRefUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setRefImage(file)
    setRefPreview(URL.createObjectURL(file))
  }

  return (
    <form onSubmit={submit} className="space-y-5">
      {/* Prompt */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Prompt</label>
        <textarea
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          rows={3} value={prompt} onChange={e => setPrompt(e.target.value)}
          placeholder="Describe the image you want to generate..."
        />
      </div>

      {/* Negative prompt */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Negative prompt (optional)</label>
        <input
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={negativePrompt} onChange={e => setNegativePrompt(e.target.value)}
          placeholder="Things to avoid..."
        />
      </div>

      {/* LoRA selector */}
      {loras.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">LoRA Adapters</label>
          <div className="space-y-2">
            {loras.map(lora => (
              <label key={lora.path} className="flex items-center gap-3 bg-gray-800 rounded-lg px-4 py-2 cursor-pointer hover:bg-gray-750">
                <input
                  type="checkbox"
                  checked={selectedLoras.includes(lora.path)}
                  onChange={() => toggleLora(lora.path)}
                  className="rounded"
                />
                <span className="flex-1 text-sm">{lora.name}</span>
                {selectedLoras.includes(lora.path) && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Scale:</span>
                    <input
                      type="range" min={0} max={1.5} step={0.05}
                      value={loraScales[lora.path] ?? 0.8}
                      onChange={e => setLoraScales(s => ({ ...s, [lora.path]: Number(e.target.value) }))}
                      className="w-20"
                    />
                    <span className="text-xs text-gray-400 w-8 text-right">{(loraScales[lora.path] ?? 0.8).toFixed(1)}</span>
                  </div>
                )}
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Parameters row */}
      <div className="flex gap-4 flex-wrap">
        <div className="w-24">
          <label className="block text-xs font-medium text-gray-400 mb-1">Steps</label>
          <input type="number" min={1} max={50} value={steps}
            onChange={e => setSteps(Number(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="w-24">
          <label className="block text-xs font-medium text-gray-400 mb-1">Guidance</label>
          <input type="number" min={0} max={20} step={0.5} value={guidance}
            onChange={e => setGuidance(Number(e.target.value))}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Reference image upload */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Reference image (img2img, optional)</label>
        <div className="flex items-center gap-4">
          <label className="bg-gray-800 border border-gray-700 border-dashed rounded-lg px-4 py-3 cursor-pointer hover:bg-gray-700 text-sm text-gray-400">
            {refImage ? refImage.name : 'Upload image…'}
            <input type="file" accept="image/*" onChange={handleRefUpload} className="hidden" />
          </label>
          {refPreview && (
            <img src={refPreview} className="h-16 w-16 object-cover rounded-lg border border-gray-700" />
          )}
          {refImage && (
            <button type="button" onClick={() => { setRefImage(null); setRefPreview(null) }}
              className="text-xs text-red-400 hover:text-red-300">Remove</button>
          )}
        </div>
      </div>

      {/* Submit */}
      <button type="submit" disabled={submitting || !prompt.trim()}
        className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
      >
        {submitting ? 'Submitting…' : 'Generate'}
      </button>

      {/* Result */}
      {resultImage && (
        <div className="mt-4">
          <img src={resultImage} alt="Generated" className="max-w-full rounded-lg border border-gray-700" />
        </div>
      )}
      {resultError && <p className="text-red-400 text-sm mt-2">Error: {resultError}</p>}
      {resultId && !resultImage && !resultError && (
        <p className="text-gray-400 text-sm mt-2">Job {resultId.slice(0, 8)}… — {submitting ? 'queued' : 'running'}</p>
      )}
    </form>
  )
}

// ═══════════════════════════════════════════════════════════════
//  Train Tab
// ═══════════════════════════════════════════════════════════════

function TrainTab() {
  const [name, setName] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const [epochs, setEpochs] = useState(50)
  const [steps, setSteps] = useState(9)
  const [training, setTraining] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const handleFiles = (newFiles: FileList | null) => {
    if (!newFiles) return
    const incoming = Array.from(newFiles)
    setFiles(prev => [...prev, ...incoming])
    for (const f of incoming) {
      setPreviews(prev => [...prev, URL.createObjectURL(f)])
    }
  }

  const removeFile = (idx: number) => {
    setFiles(prev => prev.filter((_, i) => i !== idx))
    setPreviews(prev => prev.filter((_, i) => i !== idx))
  }

  const startTraining = async () => {
    if (!name.trim() || files.length < 3) return
    setTraining(true)
    setJobId(null)
    setStatus('Encoding images…')

    try {
      const images_b64 = await Promise.all(files.map(readFileAsBase64))
      setStatus('Submitting training job…')

      const res = await fetch(`${INFERENCE_API}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), images: images_b64, num_epochs: epochs, steps }),
      })
      const data = await res.json()
      setJobId(data.job_id)
      setStatus('Training started — this may take 20–60 minutes…')
      startPolling(data.job_id)
    } catch {
      setStatus('Failed to submit training job')
      setTraining(false)
    }
  }

  const startPolling = (id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${INFERENCE_API}/jobs/${id}`)
        const data = await res.json()
        if (data.status === 'completed') {
          setStatus('✅ Training complete! The LoRA is now available in the Generate tab.')
          if (pollRef.current) clearInterval(pollRef.current)
          setTraining(false)
          // Refresh LoRA list
        } else if (data.status === 'failed') {
          setStatus(`❌ Training failed: ${data.error || 'Unknown error'}`)
          if (pollRef.current) clearInterval(pollRef.current)
          setTraining(false)
        } else {
          setStatus(`⏳ Training in progress… (${data.status})`)
        }
      } catch {
        // keep polling
      }
    }, 5000)
  }

  return (
    <div className="space-y-5">
      <div className="bg-blue-900/20 border border-blue-800 rounded-lg p-4 text-sm text-blue-200">
        <strong>Tip:</strong> 10–30 high-quality images produce the best style LoRA.
        Each image should be consistent in style but varied in content.
      </div>

      {/* Name */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">LoRA name</label>
        <input
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={name} onChange={e => setName(e.target.value)}
          placeholder="e.g. my-art-style"
          disabled={training}
        />
      </div>

      {/* Image upload */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Training images ({files.length} selected)
        </label>
        <div
          className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center cursor-pointer hover:border-gray-500 transition-colors"
          onClick={() => document.getElementById('train-file-input')?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); handleFiles(e.dataTransfer.files) }}
        >
          <p className="text-gray-400 text-sm">Drop images here, or click to browse</p>
          <input id="train-file-input" type="file" accept="image/*" multiple
            onChange={e => handleFiles(e.target.files)} className="hidden" disabled={training}
          />
        </div>
        {previews.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {previews.map((url, i) => (
              <div key={i} className="relative group">
                <img src={url} className="h-20 w-20 object-cover rounded-lg border border-gray-700" />
                {!training && (
                  <button onClick={() => removeFile(i)}
                    className="absolute -top-1.5 -right-1.5 bg-red-600 text-white text-xs rounded-full w-5 h-5 opacity-0 group-hover:opacity-100 transition-opacity"
                  >×</button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Params */}
      <div className="flex gap-4">
        <div className="w-24">
          <label className="block text-xs font-medium text-gray-400 mb-1">Epochs</label>
          <input type="number" min={1} max={500} value={epochs}
            onChange={e => setEpochs(Number(e.target.value))} disabled={training}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="w-24">
          <label className="block text-xs font-medium text-gray-400 mb-1">Steps</label>
          <input type="number" min={1} max={50} value={steps}
            onChange={e => setSteps(Number(e.target.value))} disabled={training}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      <button onClick={startTraining} disabled={training || !name.trim() || files.length < 3}
        className="bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
      >
        {training ? 'Training…' : `Start Training (${files.length} images)`}
      </button>

      {status && <p className="text-sm text-gray-300 mt-2">{status}</p>}
    </div>
  )
}

// ═══════════════════════════════════════════════════════════════
//  History Tab
// ═══════════════════════════════════════════════════════════════

function HistoryTab() {
  const [jobs, setJobs] = useState<Job[]>([])

  useEffect(() => {
    // Load from local state or session
    const stored = sessionStorage.getItem('mflux_jobs')
    if (stored) setJobs(JSON.parse(stored))

    // Fetch/reconcile with inference service (last 50)
    fetch(`${INFERENCE_API}/jobs/???`)
      .catch(() => {})
  }, [])

  const statusBadge = (status: string) => {
    const colors: Record<string, string> = {
      queued: 'bg-gray-500', running: 'bg-blue-500',
      completed: 'bg-green-500', failed: 'bg-red-500',
    }
    return (
      <span className={`inline-block px-2 py-0.5 text-xs text-white rounded ${colors[status] || 'bg-gray-500'}`}>
        {status}
      </span>
    )
  }

  if (jobs.length === 0) {
    return <p className="text-gray-500 text-sm">No generation history yet.</p>
  }

  return (
    <ul className="space-y-3">
      {jobs.map(job => (
        <li key={job.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-300 truncate">{job.prompt}</p>
              <p className="text-xs text-gray-500 mt-1">{new Date(job.createdAt).toLocaleString()}</p>
            </div>
            <div className="flex items-center gap-2 shrink-0">{statusBadge(job.status)}</div>
          </div>
          {job.resultUrl && (
            <img src={`${INFERENCE_API}/jobs/${job.id}/image?t=${Date.now()}`} alt={job.prompt}
              className="mt-3 max-w-full rounded-lg border border-gray-700 max-h-96" />
          )}
          {job.error && <p className="text-red-400 text-sm mt-2">Error: {job.error}</p>}
        </li>
      ))}
    </ul>
  )
}
