import { useState } from 'react'
import { queryApi } from './api'
import type { QueryResponse } from './types'

const EXAMPLES = [
  'What does the exp claim of a JWT control?',
  'What attack does the OAuth state parameter prevent?',
  'Which header wins: Transfer-Encoding or Content-Length?',
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [provider, setProvider] = useState('extractive')
  const [mode, setMode] = useState('hybrid')
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [active, setActive] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function ask(q?: string) {
    const text = (q ?? question).trim()
    if (text.length < 3 || loading) return
    setLoading(true)
    setError(null)
    setActive(null)
    try {
      setResult(await queryApi(text, provider, mode))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-6 flex items-baseline gap-3">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Grounded</h1>
        <p className="text-sm text-slate-500">document Q&amp;A that cites its sources</p>
      </header>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && ask()}
            placeholder="Ask something answered by the indexed documents…"
            className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
          />
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="rounded-lg border border-slate-300 px-2 py-2 text-sm text-slate-600"
            title="Retrieval mode"
          >
            <option value="hybrid">hybrid</option>
            <option value="bm25">bm25</option>
            <option value="dense">dense</option>
          </select>
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value)}
            className="rounded-lg border border-slate-300 px-2 py-2 text-sm text-slate-600"
            title="Answer provider"
          >
            <option value="extractive">extractive (no key)</option>
            <option value="openai">openai</option>
            <option value="anthropic">anthropic</option>
            <option value="ollama">ollama</option>
          </select>
          <button
            onClick={() => ask()}
            disabled={loading}
            className="rounded-lg bg-indigo-600 px-5 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            {loading ? 'Searching…' : 'Ask'}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => {
                setQuestion(example)
                void ask(example)
              }}
              className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-500 hover:border-indigo-300 hover:text-indigo-600"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <section>
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                Answer <span className="font-normal normal-case">({result.provider})</span>
              </h2>
              <GroundednessBadge value={result.groundedness} />
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              {result.sentences.length === 0 && (
                <p className="text-sm text-slate-600">{result.answer}</p>
              )}
              <div className="space-y-3">
                {result.sentences.map((sentence, i) => (
                  <div key={i} className="text-sm leading-relaxed text-slate-800">
                    <span>{sentence.text.replace(/\[\d+\]/g, '').trim()} </span>
                    {sentence.citations.map((cite) => (
                      <button
                        key={cite}
                        onClick={() => setActive(cite)}
                        className={`ml-1 rounded px-1.5 py-0.5 text-xs font-semibold ${
                          active === cite
                            ? 'bg-indigo-600 text-white'
                            : 'bg-indigo-50 text-indigo-600 hover:bg-indigo-100'
                        }`}
                      >
                        {cite}
                      </button>
                    ))}
                    {!sentence.supported && (
                      <span className="ml-2 rounded bg-amber-50 px-1.5 py-0.5 text-xs text-amber-700">
                        weak support {Math.round(sentence.support * 100)}%
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Retrieved passages
            </h2>
            <div className="space-y-3">
              {result.chunks.map((chunk) => (
                <div
                  key={chunk.position}
                  className={`rounded-xl border bg-white p-3 shadow-sm transition ${
                    active === chunk.position
                      ? 'border-indigo-400 ring-2 ring-indigo-100'
                      : 'border-slate-200'
                  }`}
                >
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span className="rounded bg-slate-100 px-1.5 py-0.5 font-semibold text-slate-600">
                      [{chunk.position}]
                    </span>
                    <span className="font-medium text-slate-700">{chunk.doc}</span>
                    <span>
                      lines {chunk.start_line}–{chunk.end_line}
                    </span>
                    <span className="ml-auto flex gap-2 font-mono">
                      {chunk.bm25_rank !== null && <span>bm25 #{chunk.bm25_rank + 1}</span>}
                      {chunk.dense_rank !== null && <span>dense #{chunk.dense_rank + 1}</span>}
                    </span>
                  </div>
                  {chunk.heading_path.length > 0 && (
                    <div className="mb-1 text-xs text-indigo-500">
                      {chunk.heading_path.join(' › ')}
                    </div>
                  )}
                  <p
                    className={`whitespace-pre-line text-xs leading-relaxed text-slate-600 ${
                      active === chunk.position ? '' : 'line-clamp-4'
                    }`}
                  >
                    {chunk.text}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      <footer className="mt-10 text-center text-xs text-slate-400">
        Hybrid retrieval (BM25 + dense, RRF) · sentence-level citations · groundedness check ·
        default corpus: five IETF RFCs
      </footer>
    </div>
  )
}

function GroundednessBadge({ value }: { value: number }) {
  const percent = Math.round(value * 100)
  const tone =
    value >= 0.8
      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
      : value >= 0.5
        ? 'bg-amber-50 text-amber-700 border-amber-200'
        : 'bg-rose-50 text-rose-700 border-rose-200'
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-medium ${tone}`}>
      grounded {percent}%
    </span>
  )
}
