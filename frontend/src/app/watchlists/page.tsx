"use client";

import { useEffect, useState, useTransition } from "react";
import Panel from "@/components/ui/Panel";
import { api, Watchlist, WatchlistListResponse } from "@/lib/api";

export default function WatchlistsPage() {
  const [data, setData] = useState<WatchlistListResponse | null>(null);
  const [editing, setEditing] = useState<Watchlist | null>(null);
  const [name, setName] = useState("");
  const [symbols, setSymbols] = useState<string[]>([]);
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  function load(signal?: AbortSignal) {
    return api.watchlists(signal).then(setData);
  }

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) setError(cause instanceof Error ? cause.message : String(cause));
    });
    return () => controller.abort();
  }, []);

  function resetForm() {
    setEditing(null);
    setName("");
    setSymbols([]);
  }

  function save() {
    setError(null);
    startTransition(async () => {
      try {
        if (editing) await api.updateWatchlist(editing.id, { name, symbols }, pin);
        else await api.createWatchlist({ name, symbols }, pin);
        await load();
        resetForm();
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    });
  }

  function remove(item: Watchlist) {
    if (!window.confirm(`Delete ${item.name}?`)) return;
    setError(null);
    startTransition(async () => {
      try {
        await api.deleteWatchlist(item.id, pin);
        await load();
        if (editing?.id === item.id) resetForm();
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    });
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Watchlists</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Persisted symbol groups for snapshot screens and alert rules. Symbols are bounded to the scheduled capture universe.
        </p>
      </div>

      <p className="rounded border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">
        Shared server workspace: every connected browser can view these lists, while changes require the server workspace PIN.
      </p>

      <label className="block max-w-xs text-xs text-muted">Workspace PIN<input type="password" value={pin} onChange={(event) => setPin(event.target.value)} maxLength={128} autoComplete="current-password" className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-foreground" /></label>

      {error && <p className="rounded border border-rose-500/50 px-3 py-2 text-sm text-rose-300">{error}</p>}

      <div className="grid gap-4 xl:grid-cols-[22rem_1fr]">
        <Panel title={editing ? `Edit ${editing.name}` : "New watchlist"}>
          <fieldset disabled={pending} className="space-y-3 disabled:opacity-70">
            <label className="block text-xs text-muted">Name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={64} className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-foreground" /></label>
            <div>
              <p className="mb-2 text-xs text-muted">Monitored symbols</p>
              <div className="flex flex-wrap gap-2">
                {(data?.available_symbols ?? []).map((symbol) => {
                  const selected = symbols.includes(symbol);
                  return <button key={symbol} type="button" aria-pressed={selected} onClick={() => setSymbols(selected ? symbols.filter((item) => item !== symbol) : [...symbols, symbol])} className={`rounded border px-2 py-1 font-mono text-xs ${selected ? "border-accent bg-accent/15 text-foreground" : "border-border text-muted"}`}>{symbol}</button>;
                })}
              </div>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={save} disabled={!pin || !name.trim() || !symbols.length} className="rounded border border-accent bg-accent/15 px-3 py-1.5 text-sm disabled:opacity-40">{pending ? "Saving..." : editing ? "Update" : "Create"}</button>
              {editing && <button type="button" onClick={resetForm} className="rounded border border-border px-3 py-1.5 text-sm text-muted">Cancel</button>}
            </div>
          </fieldset>
        </Panel>

        <Panel title="Saved watchlists" right={<span className="font-mono text-xs text-muted">{data?.items.length ?? 0} lists</span>} bodyClassName="p-0">
          <div className="divide-y divide-border">
            {(data?.items ?? []).map((item) => (
              <div key={item.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                <div><p className="font-medium text-foreground">{item.name}</p><div className="mt-1 flex flex-wrap gap-1">{item.symbols.map((symbol) => <span key={symbol} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-muted">{symbol}</span>)}</div></div>
                <div className="flex gap-2"><button disabled={pending} onClick={() => { setEditing(item); setName(item.name); setSymbols(item.symbols); }} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40">Edit</button><button disabled={pending || !pin} onClick={() => remove(item)} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative disabled:opacity-40">Delete</button></div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
