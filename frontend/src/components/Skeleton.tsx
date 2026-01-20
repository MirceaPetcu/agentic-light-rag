interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse bg-[hsl(var(--muted))] rounded ${className}`}
    />
  );
}

export function ResponseSkeleton() {
  return (
    <div className="space-y-4">
      <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-6">
        <div className="flex items-center justify-between mb-4">
          <Skeleton className="h-6 w-32" />
          <div className="flex gap-2">
            <Skeleton className="h-6 w-20" />
            <Skeleton className="h-6 w-16" />
          </div>
        </div>
        <div className="space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
        </div>
      </div>

      <div className="bg-[hsl(var(--card))] rounded-xl border border-[hsl(var(--border))] p-6">
        <Skeleton className="h-6 w-40 mb-4" />
        <div className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-full" />
        </div>
      </div>
    </div>
  );
}
