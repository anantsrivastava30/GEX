"use client";

import { useEffect, useId, useState } from "react";

export type HelpSection = {
  heading?: string;
  bullets: readonly string[];
};

export default function PageHelp({
  label,
  sections,
}: {
  label: string;
  sections: readonly HelpSection[];
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open]);

  return (
    <div className="relative shrink-0">
      <button
        type="button"
        aria-label={`How to use ${label}`}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((current) => !current)}
        className={`grid h-7 w-7 place-items-center rounded-full border text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
          open
            ? "border-accent bg-accent/15 text-accent"
            : "border-border bg-surface text-muted hover:bg-surface-hover hover:text-foreground"
        }`}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          className="h-4 w-4"
          aria-hidden
        >
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v6" />
          <path d="M12 7h.01" />
        </svg>
      </button>

      {open && (
        <>
          <button
            type="button"
            aria-label="Close instructions"
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 cursor-default bg-background/70 backdrop-blur-[1px] sm:bg-transparent sm:backdrop-blur-none"
          />
          <section
            id={panelId}
            role="dialog"
            aria-modal="true"
            aria-label={`How to use ${label}`}
            className="fixed inset-x-3 top-16 z-50 max-h-[calc(100vh-5rem)] overflow-y-auto rounded-lg border border-border bg-surface p-4 shadow-2xl sm:absolute sm:inset-x-auto sm:right-0 sm:top-9 sm:max-h-[min(38rem,calc(100vh-5rem))] sm:w-[26rem]"
          >
            <div className="mb-3 flex items-center justify-between gap-3 border-b border-border pb-2">
              <h2 className="text-sm font-semibold text-foreground">
                How to use {label}
              </h2>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded px-2 py-1 text-xs text-muted hover:bg-surface-hover hover:text-foreground"
              >
                Close
              </button>
            </div>
            <div className="space-y-4">
              {sections.map((section, index) => (
                <div key={section.heading ?? index}>
                  {section.heading && (
                    <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-accent">
                      {section.heading}
                    </h3>
                  )}
                  <ul className="space-y-1.5 text-sm leading-5 text-muted">
                    {section.bullets.map((bullet) => (
                      <li key={bullet} className="flex gap-2">
                        <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-accent" />
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
