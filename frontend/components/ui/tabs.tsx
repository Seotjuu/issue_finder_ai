"use client";

import { createContext, useContext, useId, useState, type HTMLAttributes, type ReactNode } from "react";

type TabsContextValue = {
  value: string;
  setValue: (value: string) => void;
  baseId: string;
};

const TabsContext = createContext<TabsContextValue | null>(null);

function cn(...classes: Array<string | undefined | false>) {
  return classes.filter(Boolean).join(" ");
}

function useTabs() {
  const context = useContext(TabsContext);

  if (!context) {
    throw new Error("Tabs components must be used inside Tabs");
  }

  return context;
}

type TabsProps = HTMLAttributes<HTMLDivElement> & {
  defaultValue: string;
  children: ReactNode;
};

export function Tabs({ defaultValue, children, className, ...props }: TabsProps) {
  const [value, setValue] = useState(defaultValue);
  const baseId = useId();

  return (
    <TabsContext.Provider value={{ value, setValue, baseId }}>
      <div className={className} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
}

export function TabsList({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("inline-flex rounded-md border border-zinc-200 bg-zinc-100 p-1", className)}
      role="tablist"
      {...props}
    />
  );
}

type TabsTriggerProps = HTMLAttributes<HTMLButtonElement> & {
  value: string;
};

export function TabsTrigger({ value, className, children, ...props }: TabsTriggerProps) {
  const tabs = useTabs();
  const selected = tabs.value === value;

  return (
    <button
      aria-controls={`${tabs.baseId}-panel-${value}`}
      aria-selected={selected}
      className={cn(
        "h-9 px-4 text-sm font-semibold text-zinc-600 transition hover:text-zinc-950",
        selected && "bg-white text-zinc-950 shadow-sm",
        className,
      )}
      id={`${tabs.baseId}-trigger-${value}`}
      onClick={() => tabs.setValue(value)}
      role="tab"
      type="button"
      {...props}
    >
      {children}
    </button>
  );
}

type TabsContentProps = HTMLAttributes<HTMLDivElement> & {
  value: string;
};

export function TabsContent({ value, className, ...props }: TabsContentProps) {
  const tabs = useTabs();

  if (tabs.value !== value) {
    return null;
  }

  return (
    <div
      aria-labelledby={`${tabs.baseId}-trigger-${value}`}
      className={cn("mt-5", className)}
      id={`${tabs.baseId}-panel-${value}`}
      role="tabpanel"
      tabIndex={0}
      {...props}
    />
  );
}
