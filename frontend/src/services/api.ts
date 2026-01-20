import type {
  HealthStatus,
  IngestRequest,
  IngestResponse,
  IngestStatus,
  QueryRequest,
  QueryResponse,
  SimpleQueryRequest,
  SimpleQueryResponse,
} from '../types';

const API_BASE = '/api';

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  health: {
    check: (): Promise<HealthStatus> => fetchApi('/health'),
  },

  ingest: {
    text: (data: IngestRequest): Promise<IngestResponse> =>
      fetchApi('/ingest', {
        method: 'POST',
        body: JSON.stringify(data),
      }),

    file: async (file: File): Promise<IngestResponse> => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/ingest/file`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new ApiError(response.status, error.detail || `HTTP ${response.status}`);
      }

      return response.json();
    },

    status: (trackId: string): Promise<IngestStatus> =>
      fetchApi(`/status/${trackId}`),
  },

  query: {
    advanced: (data: QueryRequest): Promise<QueryResponse> =>
      fetchApi('/query', {
        method: 'POST',
        body: JSON.stringify(data),
      }),

    simple: (data: SimpleQueryRequest): Promise<SimpleQueryResponse> =>
      fetchApi('/query/simple', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
  },
};

export { ApiError };
