import { CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface StatusIndicatorProps {
  status: 'idle' | 'loading' | 'success' | 'error';
  label?: string;
  message?: string;
}

export default function StatusIndicator({ status, label, message }: StatusIndicatorProps) {
  const statusConfig = {
    idle: {
      icon: null,
      color: 'text-[hsl(var(--muted-foreground))]',
      bgColor: 'bg-[hsl(var(--muted))]',
    },
    loading: {
      icon: <Loader2 className="w-5 h-5 animate-spin" />,
      color: 'text-[hsl(var(--primary))]',
      bgColor: 'bg-[hsl(var(--primary)/0.1)]',
    },
    success: {
      icon: <CheckCircle className="w-5 h-5" />,
      color: 'text-[hsl(var(--success))]',
      bgColor: 'bg-[hsl(var(--success)/0.1)]',
    },
    error: {
      icon: <XCircle className="w-5 h-5" />,
      color: 'text-[hsl(var(--destructive))]',
      bgColor: 'bg-[hsl(var(--destructive)/0.1)]',
    },
  };

  const config = statusConfig[status];

  if (status === 'idle' && !label) return null;

  return (
    <div className={`flex items-center gap-3 p-4 rounded-lg ${config.bgColor}`}>
      {config.icon && <span className={config.color}>{config.icon}</span>}
      <div>
        {label && <p className={`font-medium ${config.color}`}>{label}</p>}
        {message && (
          <p className="text-sm text-[hsl(var(--muted-foreground))] mt-0.5">{message}</p>
        )}
      </div>
    </div>
  );
}
