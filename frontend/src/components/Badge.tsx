import type { HTMLAttributes, ReactNode } from 'react';

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'destructive' | 'outline';
  children: ReactNode;
}

export default function Badge({ className = '', variant = 'default', children, ...props }: BadgeProps) {
  const variants = {
    default: 'bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))]',
    success: 'bg-[hsl(var(--success))] text-white',
    warning: 'bg-[hsl(var(--warning))] text-white',
    destructive: 'bg-[hsl(var(--destructive))] text-white',
    outline: 'bg-transparent border border-[hsl(var(--border))] text-[hsl(var(--foreground))]',
  };

  return (
    <span
      className={`
        inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
        ${variants[variant]}
        ${className}
      `}
      {...props}
    >
      {children}
    </span>
  );
}
