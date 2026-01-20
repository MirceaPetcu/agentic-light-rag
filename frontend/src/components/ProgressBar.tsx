interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'success' | 'warning';
}

export default function ProgressBar({
  value,
  max = 100,
  label,
  showPercentage = true,
  size = 'md',
  variant = 'default',
}: ProgressBarProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));

  const sizes = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  const variants = {
    default: 'bg-[hsl(var(--primary))]',
    success: 'bg-[hsl(var(--success))]',
    warning: 'bg-[hsl(var(--warning))]',
  };

  return (
    <div className="w-full">
      {(label || showPercentage) && (
        <div className="flex justify-between items-center mb-2">
          {label && (
            <span className="text-sm font-medium text-[hsl(var(--foreground))]">{label}</span>
          )}
          {showPercentage && (
            <span className="text-sm text-[hsl(var(--muted-foreground))]">
              {Math.round(percentage)}%
            </span>
          )}
        </div>
      )}
      <div className={`w-full rounded-full bg-[hsl(var(--muted))] overflow-hidden ${sizes[size]}`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${variants[variant]}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
