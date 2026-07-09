"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function TopBar() {
  const router = useRouter();
  const [query, setQuery] = useState("");

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const symbol = query.trim().toUpperCase();
    if (symbol) {
      router.push(`/stock/${symbol}`);
      setQuery("");
    }
  }

  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-4 border-b border-border bg-background/80 px-6 backdrop-blur">
      <form onSubmit={submit} className="flex-1 max-w-sm">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search ticker… (e.g. SPY)"
          className="w-full rounded-md border border-border bg-surface px-3 py-1.5 text-sm placeholder:text-muted focus:border-accent focus:outline-none"
        />
      </form>
      <span className="ml-auto text-xs text-muted">
        Dealer positioning, explained.
      </span>
    </header>
  );
}
