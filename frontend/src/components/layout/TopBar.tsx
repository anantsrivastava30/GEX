"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { IconSearch } from "./icons";

// US equity regular session in Eastern time, ignoring holidays.
function marketStatus(now: Date) {
  const et = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" }),
  );
  const day = et.getDay();
  const minutes = et.getHours() * 60 + et.getMinutes();
  const open = day >= 1 && day <= 5 && minutes >= 570 && minutes < 960;
  const label = et.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "America/New_York",
  });
  return { open, label };
}

export default function TopBar() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [now, setNow] = useState<Date | null>(null);

  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const symbol = query.trim().toUpperCase();
    if (symbol) {
      router.push(`/stock/${symbol}`);
      setQuery("");
    }
  }

  const status = now ? marketStatus(now) : null;

  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-4 border-b border-border bg-background/80 px-6 backdrop-blur">
      <form onSubmit={submit} className="relative w-full max-w-sm">
        <IconSearch className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search ticker… (e.g. SPY)"
          className="w-full rounded-md border border-border bg-surface py-1.5 pl-9 pr-3 text-sm placeholder:text-faint focus:border-accent focus:outline-none"
        />
      </form>

      {status && (
        <div className="ml-auto flex items-center gap-2 text-xs text-muted">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              status.open ? "bg-positive" : "bg-faint"
            }`}
          />
          <span>{status.open ? "Market open" : "Market closed"}</span>
          <span className="font-mono text-faint">{status.label} ET</span>
        </div>
      )}
    </header>
  );
}
