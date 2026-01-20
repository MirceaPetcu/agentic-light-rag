import { createContext, useContext, useState, type ReactNode } from 'react';

interface TabsContextValue {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

interface TabsProps {
  defaultValue: string;
  children: ReactNode;
  className?: string;
}

export function Tabs({ defaultValue, children, className = '' }: TabsProps) {
  const [activeTab, setActiveTab] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
}

interface TabsListProps {
  children: ReactNode;
  className?: string;
}

export function TabsList({ children, className = '' }: TabsListProps) {
  return (
    <div
      className={`
        inline-flex items-center p-1 rounded-lg
        bg-[hsl(var(--muted))]
        ${className}
      `}
    >
      {children}
    </div>
  );
}

interface TabsTriggerProps {
  value: string;
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}

export function TabsTrigger({ value, children, className = '', onClick }: TabsTriggerProps) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabsTrigger must be used within Tabs');

  const isActive = context.activeTab === value;

  const handleClick = () => {
    context.setActiveTab(value);
    onClick?.();
  };

  return (
    <button
      onClick={handleClick}
      className={`
        px-4 py-2 text-sm font-medium rounded-md transition-all duration-200
        ${
          isActive
            ? 'bg-[hsl(var(--card))] text-[hsl(var(--foreground))] shadow-sm'
            : 'text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]'
        }
        ${className}
      `}
    >
      {children}
    </button>
  );
}

interface TabsContentProps {
  value: string;
  children: ReactNode;
  className?: string;
}

export function TabsContent({ value, children, className = '' }: TabsContentProps) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabsContent must be used within Tabs');

  if (context.activeTab !== value) return null;

  return <div className={`animate-fade-in ${className}`}>{children}</div>;
}
