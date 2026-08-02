"use client";

// Underlined tab strip, UW-style. Controlled by the parent so pages can
// swap content without routing.

export interface Tab {
  id: string;
  label: string;
  soon?: boolean;
}

export default function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="flex items-center gap-1 border-b border-border">
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            onClick={() => !tab.soon && onChange(tab.id)}
            disabled={tab.soon}
            className={`relative px-3 py-2 text-sm transition-colors ${
              isActive
                ? "text-foreground"
                : tab.soon
                  ? "cursor-not-allowed text-faint"
                  : "text-muted hover:text-foreground"
            }`}
          >
            {tab.label}
            {tab.soon && (
              <span className="ml-1.5 rounded bg-border px-1 py-0.5 text-[9px] uppercase text-faint">
                soon
              </span>
            )}
            {isActive && (
              <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-accent" />
            )}
          </button>
        );
      })}
    </div>
  );
}
