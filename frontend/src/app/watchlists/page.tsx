"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import Panel from "@/components/ui/Panel";
import { api, Watchlist, WatchlistListResponse } from "@/lib/api";

export default function WatchlistsPage() {
  const loadSequence = useRef(0);
  const [data, setData] = useState<WatchlistListResponse | null>(null);
  const [editing, setEditing] = useState<Watchlist | null>(null);
  const [name, setName] = useState("");
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbolInput, setSymbolInput] = useState("");
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, startTransition] = useTransition();

  function load(signal?: AbortSignal) {
    const sequence = ++loadSequence.current;
    return api.watchlists(signal).then((response) => {
      if (loadSequence.current === sequence) setData(response);
    }).finally(() => {
      if (loadSequence.current === sequence) setLoading(false);
    });
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
    setSymbolInput("");
  }

  function addSymbols() {
    const candidates = symbolInput
      .toUpperCase()
      .split(/[\s,]+/)
      .filter(Boolean);
    const invalid = candidates.filter((symbol) => !/^[A-Z][A-Z0-9.-]{0,9}$/.test(symbol));
    if (invalid.length) {
      setError(`Invalid ticker format: ${invalid.join(", ")}`);
      return;
    }
    const next = [...new Set([...symbols, ...candidates])];
    if (next.length > 25) {
      setError("A watchlist can contain at most 25 symbols.");
      return;
    }
    setSymbols(next);
    setSymbolInput("");
    setError(null);
  }

  function save() {
    setError(null);
    startTransition(async () => {
      try {
        if (editing) await api.updateWatchlist(editing.id, { name, symbols }, pin);
        else await api.createWatchlist({ name, symbols }, pin);
        setLoading(true);
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
          Persisted symbol groups for snapshot screens and alert rules. Add any ticker-shaped symbol; custom entries join the server&apos;s scheduled capture universe.
        </p>
      </div>

      <p className="rounded border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">
        Shared server workspace: every connected browser can view these lists, while changes require the server workspace PIN.
      </p>

      <div className="rounded border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-100">
        <p className="font-medium">Watchlist changes affect future snapshots.</p>
        <p className="mt-1 text-amber-200/80">
          A custom symbol starts capturing on the next enabled server run. Current-chain views may work immediately, but change metrics require two consecutive market sessions and rank or history views become more useful as observations accumulate. Removing a custom symbol from every watchlist stops new captures without deleting saved history; baseline symbols continue to run. An unrecognized or non-optionable ticker remains unavailable and can cause alerts tied to that watchlist to skip until it is corrected.
        </p>
      </div>

      {data && !data.scheduler_enabled && <p className="rounded border border-rose-500/40 bg-rose-500/5 px-3 py-2 text-xs text-rose-200">The server snapshot scheduler is disabled. Custom symbols can be saved, but they will not accrue snapshots until scheduled capture is enabled.</p>}
      {data?.scheduler_enabled && !data.tradier_configured && <p className="rounded border border-rose-500/40 bg-rose-500/5 px-3 py-2 text-xs text-rose-200">The snapshot scheduler is enabled, but Tradier credentials are missing. No symbols will accrue snapshots until `TRADIER_TOKEN` is configured.</p>}

      <label className="block max-w-xs text-xs text-muted">Workspace PIN<input type="password" value={pin} onChange={(event) => setPin(event.target.value)} maxLength={128} autoComplete="current-password" className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-foreground" /></label>

      {error && <p className="rounded border border-rose-500/50 px-3 py-2 text-sm text-rose-300">{error}</p>}

      <div className="grid gap-4 xl:grid-cols-[22rem_1fr]">
        <Panel title={editing ? `Edit ${editing.name}` : "New watchlist"}>
          <fieldset disabled={pending || loading} className="space-y-3 disabled:opacity-70">
            <label className="block text-xs text-muted">Name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={64} className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-foreground" /></label>
            <div>
              <div className="mb-2 flex items-center justify-between gap-2 text-xs text-muted">
                <span>Monitored symbols</span>
                <span className="font-mono">{loading ? "... scheduled" : `${data?.available_symbols.length ?? 0}/${data?.snapshot_symbol_limit ?? 50} scheduled`}</span>
              </div>
              <div className="flex gap-2">
                <input value={symbolInput} onChange={(event) => setSymbolInput(event.target.value.toUpperCase())} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); addSymbols(); } }} maxLength={128} placeholder="INTC, MU, SMH" aria-label="Tickers to add" className="min-w-0 flex-1 rounded border border-border bg-surface-2 px-2 py-1.5 font-mono text-sm uppercase text-foreground" />
                <button type="button" onClick={addSymbols} disabled={!symbolInput.trim()} className="rounded border border-border px-3 py-1.5 text-xs text-muted hover:text-foreground disabled:opacity-40">Add</button>
              </div>
              <p className="mt-1 text-[11px] text-muted">Enter one or more symbols separated by commas or spaces.</p>
              <div className="mt-3 flex min-h-8 flex-wrap gap-2 rounded border border-border bg-surface px-2 py-2">
                {symbols.map((symbol) => <span key={symbol} className="inline-flex items-center gap-1 rounded border border-accent/50 bg-accent/10 px-2 py-1 font-mono text-xs text-foreground">{symbol}<button type="button" onClick={() => setSymbols(symbols.filter((item) => item !== symbol))} aria-label={`Remove ${symbol}`} className="text-muted hover:text-negative">&times;</button></span>)}
                {!symbols.length && <span className="text-xs text-muted">No symbols selected.</span>}
              </div>
              <p className="mb-2 mt-3 text-[11px] uppercase tracking-wide text-muted">Currently scheduled</p>
              <div className="flex max-h-28 flex-wrap gap-2 overflow-auto">
                {loading && <span className="text-xs text-muted">Loading scheduled symbols...</span>}
                {(data?.available_symbols ?? []).filter((symbol) => !symbols.includes(symbol)).map((symbol) => <button key={symbol} type="button" onClick={() => { if (symbols.length >= 25) { setError("A watchlist can contain at most 25 symbols."); return; } setSymbols([...symbols, symbol]); setError(null); }} className="rounded border border-border px-2 py-1 font-mono text-xs text-muted hover:border-accent hover:text-foreground">+ {symbol}</button>)}
              </div>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={save} disabled={!pin || !name.trim() || !symbols.length} className="rounded border border-accent bg-accent/15 px-3 py-1.5 text-sm disabled:opacity-40">{pending ? "Saving..." : editing ? "Update" : "Create"}</button>
              {editing && <button type="button" onClick={resetForm} className="rounded border border-border px-3 py-1.5 text-sm text-muted">Cancel</button>}
            </div>
          </fieldset>
        </Panel>

        <Panel title="Saved watchlists" right={<span className="font-mono text-xs text-muted">{loading ? "..." : `${data?.items.length ?? 0} lists`}</span>} bodyClassName="p-0">
          <div className="divide-y divide-border">
            {loading && <p className="px-4 py-10 text-center text-sm text-muted">Loading watchlists...</p>}
            {(data?.items ?? []).map((item) => (
              <div key={item.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                <div><p className="font-medium text-foreground">{item.name}</p><div className="mt-1 flex flex-wrap gap-1">{item.symbols.map((symbol) => <span key={symbol} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-muted">{symbol}</span>)}</div></div>
                <div className="flex gap-2"><button disabled={pending} onClick={() => { setEditing(item); setName(item.name); setSymbols(item.symbols); setSymbolInput(""); }} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40">Edit</button><button disabled={pending || !pin} onClick={() => remove(item)} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative disabled:opacity-40">Delete</button></div>
              </div>
            ))}
            {!loading && !error && data && !data.items.length && <p className="px-4 py-10 text-center text-sm text-muted">No saved watchlists yet. Create one using the form.</p>}
          </div>
        </Panel>
      </div>
    </div>
  );
}
