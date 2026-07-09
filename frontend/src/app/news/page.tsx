"use client";

import { useEffect, useState } from "react";

interface NewsItem {
  title: string;
  link: string;
  source: string;
  date: string;
}

export default function NewsPage() {
  const [items, setItems] = useState<NewsItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/news")
      .then((resp) => {
        if (!resp.ok) throw new Error(`Backend returned ${resp.status}`);
        return resp.json();
      })
      .then(setItems)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Market News</h1>
      {error && <p className="text-sm text-muted">{error}</p>}
      {!items && !error && <p className="text-sm text-muted">Loading…</p>}
      {items && (
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
