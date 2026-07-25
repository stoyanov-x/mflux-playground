import { useState, useRef, FormEvent } from 'react'

type Job = {
  id: string
  prompt: string
  status: string
  resultUrl?: string
  error?: string
  createdAt: string
}

const API = '/api/generation'

function App() {
  const [prompt, setPrompt] = useState('')
  const [loraScale, setLoraScale] = useState(0.8)
  const [steps, setSteps] = useState(25)
  const [submitting, setSubmitting] = useState(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const pollRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!prompt.trim()) return

    setSubmitting(true)
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          loraScales: [loraScale],
          numInferenceSteps: steps,
        }),
      })
      const data = await res.json()
      const job: Job = {
        id: data.id,
        prompt: prompt.trim(),
        status: 'queued',
        createdAt: new Date().toISOString(),
      }
      setJobs(prev => [job, ...prev])
      startPolling(data.id)
    } catch (err) {
      console.error('Submit failed', err)
    } finally {
      setSubmitting(false)
    }
  }

  const startPolling = (id: string) => {
    pollRef.current[id] = setInterval(async () => {
      try {
        const res = await fetch(`${API}/${id}`)
        const data = await res.json()
        setJobs(prev =>
          prev.map(j => (j.id === id ? { ...j, status: data.status, resultUrl: data.resultUrl, error: data.error } : j))
        )
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollRef.current[id])
          delete pollRef.current[id]
        }
      } catch {
        clearInterval(pollRef.current[id])
        delete pollRef.current[id]
      }
    }, 2000)
  }

  const statusBadge = (status: string) => {
    const colors: Record<string, string> = {
      queued: 'bg-gray-500',
      running: 'bg-blue-500',
      completed: 'bg-green-500',
      failed: 'bg-red-500',
    }
    return (
      <span className={`inline-block px-2 py-0.5 text-xs text-white rounded ${colors[status] || 'bg-gray-500'}`}>
        {status}
      </span>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-3xl mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">mflux</h1>
          <p className="text-gray-400 mt-1">Flux image generation on Apple Silicon</p>
        </header>

        <form onSubmit={submit} className="space-y-4 mb-12">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Prompt</label>
            <textarea
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              rows={3}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Describe the image you want to generate..."
            />
          </div>

          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-300 mb-1">
                LoRA Scale ({loraScale.toFixed(1)})
              </label>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={loraScale}
                onChange={e => setLoraScale(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div className="w-24">
              <label className="block text-sm font-medium text-gray-300 mb-1">Steps</label>
              <input
                type="number"
                min={1}
                max={50}
                value={steps}
                onChange={e => setSteps(Number(e.target.value))}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting || !prompt.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
          >
            {submitting ? 'Submitting…' : 'Generate'}
          </button>
        </form>

        <section>
          <h2 className="text-xl font-semibold mb-4">Generation History</h2>
          {jobs.length === 0 ? (
            <p className="text-gray-500">No jobs yet — submit a prompt above.</p>
          ) : (
            <ul className="space-y-3">
              {jobs.map(job => (
                <li key={job.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-gray-300 truncate">{job.prompt}</p>
                      <p className="text-xs text-gray-500 mt-1">{new Date(job.createdAt).toLocaleString()}</p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {statusBadge(job.status)}
                    </div>
                  </div>
                  {job.resultUrl && (
                    <div className="mt-3">
                      <img
                        src={`${API}/${job.id}/image`}
                        alt={job.prompt}
                        className="max-w-full rounded-lg border border-gray-700"
                      />
                    </div>
                  )}
                  {job.error && (
                    <p className="text-red-400 text-sm mt-2">Error: {job.error}</p>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
