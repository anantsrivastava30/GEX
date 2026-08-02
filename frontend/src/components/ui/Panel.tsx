import { ReactNode } from "react";

// Bordered surface card with an optional titled header row — the base
// container for every dense block in the terminal.

export default function Panel({
  title,
  right,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`rounded-lg border border-border bg-surface ${className}`}
    >
      {(title || right) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5">
          <h2 className="text-xs font-medium uppercase tracking-wide text-muted">
            {title}
          </h2>
          {right}
        </header>
      )}
      <div className={`p-4 ${bodyClassName}`}>{children}</div>
    </section>
  );
}
