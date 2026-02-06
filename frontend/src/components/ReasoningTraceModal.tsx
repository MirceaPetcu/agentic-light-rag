import { useEffect, useState } from 'react';
import { X, Loader2, Brain } from 'lucide-react';
import { api, ApiError } from '../services/api';

interface ReasoningTraceModalProps {
  queryId: string;
  onClose: () => void;
}

export default function ReasoningTraceModal({ queryId, onClose }: ReasoningTraceModalProps) {
  const [trace, setTrace] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api.query.reasoningTrace(queryId)
      .then((res) => {
        if (!cancelled) setTrace(res.reasoning_trace);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : 'Failed to load reasoning trace');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [queryId]);

  return (
    <div className="citation-overlay" onClick={onClose}>
      <div className="reasoning-trace-modal" onClick={(e) => e.stopPropagation()}>
        <div className="citation-modal-header">
          <div className="citation-modal-title">
            <Brain className="icon" />
            <span>Reasoning Trace</span>
          </div>
          <button className="citation-modal-close" onClick={onClose}>
            <X className="icon" />
          </button>
        </div>
        <div className="citation-modal-body">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-5 h-5 animate-spin text-[hsl(var(--muted-foreground))]" />
            </div>
          )}
          {error && (
            <p className="text-[hsl(var(--destructive))] text-sm">{error}</p>
          )}
          {trace && (
            <pre className="reasoning-trace-content">{JSON.stringify(trace, null, 2)}</pre>
          )}
        </div>
      </div>
    </div>
  );
}
