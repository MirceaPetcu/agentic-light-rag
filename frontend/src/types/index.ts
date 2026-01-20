export interface HealthStatus {
  status: string;
  rag_initialized: boolean;
  redis_connected: boolean;
}

export interface IngestRequest {
  content: string;
  doc_id?: string;
  file_path?: string;
}

export interface IngestResponse {
  status: string;
  message: string;
  track_id: string;
  doc_id?: string;
}

export interface IngestStatus {
  track_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message?: string;
  progress?: number;
}

export interface QueryRequest {
  query: string;
  max_steps?: number;
  similarity_threshold?: number;
  top_k?: number;
}

export interface Citation {
  reference_id: string;
  source: string;
  content: string;
  file_path?: string;
}

export interface QueryMetadata {
  original_query: string;
  final_similarity_score: number;
  subqueries_used: string[];
  key_points: string[];
  limitations: string[];
}

export interface QueryResponse {
  status: string;
  response: string;
  citations: Citation[];
  confidence: number;
  steps_taken: number;
  converged: boolean;
  metadata: QueryMetadata;
}

export interface SimpleQueryRequest {
  query: string;
  mode?: 'naive' | 'local' | 'global' | 'hybrid';
}

export interface SimpleQueryResponse {
  status: string;
  response: string;
  mode: string;
}
