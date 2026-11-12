export interface ChunkOut {
  position: number
  doc: string
  heading_path: string[]
  start_line: number
  end_line: number
  text: string
  rrf_score: number
  bm25_rank: number | null
  dense_rank: number | null
  dense_similarity: number | null
}

export interface SentenceOut {
  text: string
  citations: number[]
  support: number
  supported: boolean
}

export interface QueryResponse {
  provider: string
  answer: string
  sentences: SentenceOut[]
  groundedness: number
  chunks: ChunkOut[]
}
