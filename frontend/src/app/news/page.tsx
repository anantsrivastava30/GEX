"use client";

import { useEffect, useState } from "react";
import Panel from "@/components/ui/Panel";

interface NewsItem {
  title: string;
  link: string;
  source: string;
  date: string;
}

export default function NewsPage() {
  const [items, setItems] = useState<NewsItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/news", { signal: controller.signal })
      .then((resp) => {
        if (!resp.ok) throw new Error(`Backend returned ${resp.status}`);
        return resp.json();
      })
      .then(setItems)
      .catch((cause) => {
        if (!(cause instanceof DOMException && cause.name === "AbortError")) setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => controller.abort();
  }, [reload]);

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Market News</h1>
      {error && <p role="alert" className="rounded border border-rose-500/50 bg-surface px-4 py-3 text-sm text-rose-300">Unable to load market news: {error} <button type="button" onClick={() => { setItems(null); setError(null); setReload((value) => value + 1); }} className="ml-2 text-foreground underline">Retry</button></p>}
      {!items && !error && <Panel title="Headlines"><p role="status" className="py-10 text-center text-sm text-muted">Loading market headlines...</p></Panel>}
      {items && items.length === 0 && <Panel title="Headlines"><p className="py-10 text-center text-sm text-muted">No market headlines are available from the configured feeds.</p></Panel>}
      {items && items.length > 0 && (
        <ul className="space-y-2">
          {items.map((item) => (
            <li
              key={item.link}
              className="rounded-lg border border-border bg-surface px-4 py-3 hover:bg-surface-hover"
            >
              <a href={item.link} target="_blank" rel="noreferrer" className="block">
                <p className="text-sm">{item.title}</p>
                <p className="mt-1 text-xs text-muted">
                  {item.source} · {item.date}
                </p>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
