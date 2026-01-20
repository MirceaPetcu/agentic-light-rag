import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Moon, Sun } from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== 'undefined') {
      return (
        localStorage.getItem('theme') === 'dark' ||
        (!localStorage.getItem('theme') &&
          window.matchMedia('(prefers-color-scheme: dark)').matches)
      );
    }
    return true;
  });
  const location = useLocation();

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  const navItems = [
    { path: '/', label: 'Query' },
    { path: '/ingest', label: 'Documents' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen flex flex-col bg-[hsl(var(--background))]">
      <header className="border-b border-[hsl(var(--border))]">
        <div className="max-w-3xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link
              to="/"
              className="text-[hsl(var(--foreground))] font-medium tracking-tight"
            >
              LightRAG
            </Link>

            <nav className="flex items-center gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    px-3 py-1.5 text-sm rounded-md transition-colors
                    ${
                      isActive(item.path)
                        ? 'text-[hsl(var(--foreground))] bg-[hsl(var(--muted))]'
                        : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
                    }
                  `}
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>

          <button
            onClick={() => setIsDark(!isDark)}
            className="p-2 rounded-md text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))] hover:bg-[hsl(var(--muted))] transition-colors"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </header>

      <main className="flex-1 py-12 px-6">
        <div className="max-w-3xl mx-auto">{children}</div>
      </main>
    </div>
  );
}
