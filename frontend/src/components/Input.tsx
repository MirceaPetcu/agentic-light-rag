import { type InputHTMLAttributes, forwardRef } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className = '', label, error, id, ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={id}
            className="block text-sm font-medium text-[hsl(var(--foreground))] mb-1.5"
          >
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={id}
          className={`
            w-full px-4 py-2.5 rounded-lg
            bg-[hsl(var(--card))]
            border border-[hsl(var(--border))]
            text-[hsl(var(--foreground))]
            placeholder:text-[hsl(var(--muted-foreground))]
            focus:outline-none focus:ring-2 focus:ring-[hsl(var(--primary))] focus:border-transparent
            transition-all duration-200
            disabled:opacity-50 disabled:cursor-not-allowed
            ${error ? 'border-[hsl(var(--destructive))] focus:ring-[hsl(var(--destructive))]' : ''}
            ${className}
          `}
          {...props}
        />
        {error && (
          <p className="mt-1.5 text-sm text-[hsl(var(--destructive))]">{error}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
