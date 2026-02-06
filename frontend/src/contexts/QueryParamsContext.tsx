import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';

export interface QueryParams {
  max_steps: number;
  similarity_threshold: number;
  top_k: number;
  stream: boolean;
}

export const DEFAULT_QUERY_PARAMS: QueryParams = {
  max_steps: 5,
  similarity_threshold: 0.75,
  top_k: 10,
  stream: false,
};

interface QueryParamsContextType {
  params: QueryParams;
  setParam: <K extends keyof QueryParams>(key: K, value: QueryParams[K]) => void;
  resetParams: () => void;
}

const STORAGE_KEY = 'query-params';

function loadParams(): QueryParams {
  if (typeof window === 'undefined') return DEFAULT_QUERY_PARAMS;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return DEFAULT_QUERY_PARAMS;
  try {
    return { ...DEFAULT_QUERY_PARAMS, ...JSON.parse(stored) };
  } catch {
    return DEFAULT_QUERY_PARAMS;
  }
}

const QueryParamsContext = createContext<QueryParamsContextType | null>(null);

export function QueryParamsProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useState<QueryParams>(() => loadParams());

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
  }, [params]);

  const setParam = useCallback(<K extends keyof QueryParams>(key: K, value: QueryParams[K]) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  }, []);

  const resetParams = useCallback(() => {
    setParams(DEFAULT_QUERY_PARAMS);
  }, []);

  return (
    <QueryParamsContext.Provider value={{ params, setParam, resetParams }}>
      {children}
    </QueryParamsContext.Provider>
  );
}

export function useQueryParams() {
  const context = useContext(QueryParamsContext);
  if (!context) {
    throw new Error('useQueryParams must be used within a QueryParamsProvider');
  }
  return context;
}
