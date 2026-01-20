import { useState } from 'react';
import { ArrowRight, Loader2, ChevronDown } from 'lucide-react';
import { api } from '../services/api';
import type { QueryResponse, SimpleQueryResponse } from '../types';

type QueryMode = 'advanced' | 'simple';

export default function QueryPage() {
  const [query, setQuery] = useState('');
  const [queryMode, setQueryMode] = useState<QueryMode>('advanced');
  const [simpleMode, setSimpleMode] = useState<'hybrid' | 'local' | 'global' | 'naive'>('hybrid');
  const [isLoading, setIsLoading] = useState(false);
  const [advancedResponse, setAdvancedResponse] = useState<QueryResponse | null>(null);
  const [simpleResponse, setSimpleResponse] = useState<SimpleQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCitations, setShowCitations] = useState(false);

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setAdvancedResponse(null);
    setSimpleResponse(null);

    try {
      if (queryMode === 'advanced') {
        const response = await api.query.advanced({ query });
        setAdvancedResponse(response);
      } else {
        const response = await api.query.simple({ query, mode: simpleMode });
        setSimpleResponse(response);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-[hsl(var(--foreground))] mb-2">
          Ask your documents
        </h1>
        <p className="text-[hsl(var(--muted-foreground))]">
          Query your knowledge base using multi-agent reasoning
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex gap-2">
          <button
            onClick={() => setQueryMode('advanced')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              queryMode === 'advanced'
                ? 'bg-[hsl(var(--foreground))] text-[hsl(var(--background))]'
                : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
            }`}
          >
            Multi-agent
          </button>
          <button
            onClick={() => setQueryMode('simple')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              queryMode === 'simple'
                ? 'bg-[hsl(var(--foreground))] text-[hsl(var(--background))]'
                : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
            }`}
          >
            Direct
          </button>
        </div>

        {queryMode === 'simple' && (
          <div className="flex gap-2">
            {(['hybrid', 'local', 'global', 'naive'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setSimpleMode(mode)}
                className={`px-2.5 py-1 text-xs rounded transition-colors ${
                  simpleMode === mode
                    ? 'bg-[hsl(var(--muted))] text-[hsl(var(--foreground))]'
                    : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        )}

        <div className="relative">
          <textarea
            placeholder="What would you like to know?"
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="
              w-full px-4 py-3 rounded-lg resize-none
              bg-[hsl(var(--muted))]
              text-[hsl(var(--foreground))]
              placeholder:text-[hsl(var(--muted-foreground))]
              border-0
              focus:outline-none focus:ring-1 focus:ring-[hsl(var(--border))]
              transition-shadow
            "
          />
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-[hsl(var(--muted-foreground))]">
            {queryMode === 'advanced' ? 'Iterative reasoning with citations' : `Direct ${simpleMode} search`}
          </span>
          <button
            onClick={handleSubmit}
            disabled={!query.trim() || isLoading}
            className="
              inline-flex items-center gap-2 px-4 py-2 rounded-lg
              bg-[hsl(var(--foreground))] text-[hsl(var(--background))]
              text-sm font-medium
              disabled:opacity-40 disabled:cursor-not-allowed
              hover:opacity-90 transition-opacity
            "
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Thinking...
              </>
            ) : (
              <>
                Submit
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-[hsl(var(--destructive)/0.1)] text-[hsl(var(--destructive))] text-sm animate-in">
          {error}
        </div>
      )}

      {advancedResponse && (
        <div className="space-y-6 animate-in">
          <div className="flex items-center gap-3 text-sm text-[hsl(var(--muted-foreground))]">
            <span
              className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs ${
                advancedResponse.converged
                  ? 'bg-[hsl(var(--success)/0.1)] text-[hsl(var(--success))]'
                  : 'bg-[hsl(var(--warning)/0.1)] text-[hsl(var(--warning))]'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${advancedResponse.converged ? 'bg-[hsl(var(--success))]' : 'bg-[hsl(var(--warning))]'}`}
              />
              {advancedResponse.converged ? 'Converged' : 'Partial'}
            </span>
            <span>{advancedResponse.steps_taken} steps</span>
            <span>{Math.round(advancedResponse.confidence * 100)}% confidence</span>
          </div>

          <div className="text-[hsl(var(--foreground))] leading-relaxed whitespace-pre-wrap">
            {advancedResponse.response}
          </div>

          {advancedResponse.metadata?.key_points && advancedResponse.metadata.key_points.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-[hsl(var(--muted-foreground))] uppercase tracking-wide">
                Key points
              </p>
              <ul className="space-y-1.5">
                {advancedResponse.metadata.key_points.map((point, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-[hsl(var(--foreground))]">
                    <span className="text-[hsl(var(--muted-foreground))] mt-1">•</span>
                    {point}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {advancedResponse.citations && advancedResponse.citations.length > 0 && (
            <div className="border-t border-[hsl(var(--border))] pt-4">
              <button
                onClick={() => setShowCitations(!showCitations)}
                className="flex items-center gap-2 text-sm text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] transition-colors"
              >
                <ChevronDown
                  className={`w-4 h-4 transition-transform ${showCitations ? 'rotate-180' : ''}`}
                />
                {advancedResponse.citations.length} sources
              </button>

              {showCitations && (
                <div className="mt-4 space-y-3">
                  {advancedResponse.citations.map((citation, i) => (
                    <div
                      key={citation.reference_id || i}
                      className="p-3 rounded-lg bg-[hsl(var(--muted))] text-sm"
                    >
                      <div className="flex items-start gap-2">
                        <span className="text-[hsl(var(--muted-foreground))] font-mono text-xs">
                          [{i + 1}]
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-[hsl(var(--foreground))] line-clamp-2">
                            {citation.content}
                          </p>
                          {citation.source && (
                            <p className="text-xs text-[hsl(var(--muted-foreground))] mt-1">
                              {citation.source}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {simpleResponse && (
        <div className="space-y-4 animate-in">
          <div className="text-xs text-[hsl(var(--muted-foreground))]">
            {simpleResponse.mode} mode
          </div>
          <div className="text-[hsl(var(--foreground))] leading-relaxed whitespace-pre-wrap">
            {simpleResponse.response}
          </div>
        </div>
      )}
    </div>
  );
}
