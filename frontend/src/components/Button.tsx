import { type ButtonHTMLAttributes, forwardRef } from 'react';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', variant = 'primary', size = 'md', loading, children, disabled, ...props }, ref) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary:
        'bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90 focus:ring-[hsl(var(--primary))]',
      secondary:
        'bg-[hsl(var(--secondary))] text-[hsl(var(--secondary-foreground))] hover:bg-[hsl(var(--muted))] focus:ring-[hsl(var(--secondary))]',
      ghost:
        'bg-transparent hover:bg-[hsl(var(--muted))] text-[hsl(var(--foreground))] focus:ring-[hsl(var(--muted))]',
      destructive:
        'bg-[hsl(var(--destructive))] text-white hover:opacity-90 focus:ring-[hsl(var(--destructive))]',
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-sm gap-1.5',
      md: 'px-4 py-2 text-sm gap-2',
      lg: 'px-6 py-3 text-base gap-2',
    };

    return (
      <button
        ref={ref}
        className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
        disabled={disabled || loading}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;
