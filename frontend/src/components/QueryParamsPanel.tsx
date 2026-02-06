import { useState } from 'react';
import { Settings, ChevronDown, RotateCcw } from 'lucide-react';
import { useQueryParams, DEFAULT_QUERY_PARAMS } from '../contexts/QueryParamsContext';

interface ParamSliderProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  onChange: (value: number) => void;
}

function ParamSlider({ label, description, value, min, max, step, defaultValue, onChange }: ParamSliderProps) {
  const isDefault = value === defaultValue;

  return (
    <div className="param-row">
      <div className="param-label-group">
        <div className="flex items-center gap-2">
          <span className="text-sm text-[hsl(var(--foreground))]">{label}</span>
          {!isDefault && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[hsl(var(--primary)/0.15)] text-[hsl(var(--primary))]">
              modified
            </span>
          )}
        </div>
        <span className="text-xs text-[hsl(var(--muted-foreground))]">{description}</span>
      </div>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="param-range"
        />
        <span className="text-sm font-mono text-[hsl(var(--foreground))] min-w-[3ch] text-right">
          {step < 1 ? value.toFixed(2) : value}
        </span>
      </div>
    </div>
  );
}

interface QueryParamsPanelProps {
  className?: string;
  compact?: boolean;
}

export default function QueryParamsPanel({ className = '', compact = false }: QueryParamsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { params, setParam, resetParams } = useQueryParams();

  return (
    <div className={`query-params-panel ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-sm text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] transition-colors"
      >
        <Settings className="w-4 h-4" />
        <span>Query parameters</span>
        <ChevronDown
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <div className={`mt-3 animate-in ${compact ? 'query-params-grid-compact' : 'query-params-grid'}`}>
          <ParamSlider
            label="Max steps"
            description="Maximum reasoning iterations"
            value={params.max_steps}
            min={1}
            max={15}
            step={1}
            defaultValue={DEFAULT_QUERY_PARAMS.max_steps}
            onChange={(v) => setParam('max_steps', v)}
          />

          <ParamSlider
            label="Similarity threshold"
            description="Convergence threshold"
            value={params.similarity_threshold}
            min={0}
            max={1}
            step={0.05}
            defaultValue={DEFAULT_QUERY_PARAMS.similarity_threshold}
            onChange={(v) => setParam('similarity_threshold', parseFloat(v.toFixed(2)))}
          />

          <ParamSlider
            label="Top K"
            description="Results to retrieve per step"
            value={params.top_k}
            min={1}
            max={50}
            step={1}
            defaultValue={DEFAULT_QUERY_PARAMS.top_k}
            onChange={(v) => setParam('top_k', v)}
          />

          <div className="param-row">
            <div className="param-label-group">
              <span className="text-sm text-[hsl(var(--foreground))]">Stream</span>
              <span className="text-xs text-[hsl(var(--muted-foreground))]">Real-time SSE events</span>
            </div>
            <button
              onClick={() => setParam('stream', !params.stream)}
              className={`param-toggle ${params.stream ? 'is-on' : ''}`}
              role="switch"
              aria-checked={params.stream}
            >
              <span className="param-toggle-thumb" />
            </button>
          </div>

          <button
            onClick={resetParams}
            className="flex items-center gap-1.5 text-xs text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] transition-colors mt-1"
          >
            <RotateCcw className="w-3 h-3" />
            Reset to defaults
          </button>
        </div>
      )}
    </div>
  );
}
