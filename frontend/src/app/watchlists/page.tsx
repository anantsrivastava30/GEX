"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import Panel from "@/components/ui/Panel";
import { api, Watchlist, WatchlistListResponse } from "@/lib/api";

const STARTER_LISTS = [
  { name: "Index pulse", symbols: ["SPY", "QQQ", "IWM", "DIA"] },
  { name: "Semiconductors", symbols: ["NVDA", "AMD", "AVGO", "INTC", "MU", "MRVL", "SMH"] },
  { name: "Mega-cap tech", symbols: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"] },
];

export default function WatchlistsPage() {
  const loadSequence = useRef(0);
  const formRef = useRef<HTMLDivElement>(null);
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

  function applyStarterList(starter: (typeof STARTER_LISTS)[number]) {
    setEditing(null);
    setName(starter.name);
    setSymbols(starter.symbols);
    setSymbolInput("");
    setError(null);
    formRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
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
          Group related symbols by the market question you want to monitor. A watchlist defines an alert&apos;s audience, but it does not trigger notifications by itself.
        </p>
      </div>

      <Panel title="How watchlists work">
        <div className="grid gap-3 text-sm md:grid-cols-3">
          <div><p className="font-medium text-foreground">1. Choose a purpose</p><p className="mt-1 text-xs text-muted">Use a focused theme such as index risk, semiconductors, or your active holdings.</p></div>
          <div><p className="font-medium text-foreground">2. Accumulate snapshots</p><p className="mt-1 text-xs text-muted">The server adds these symbols to scheduled option-chain captures. New symbols become useful after their first capture.</p></div>
          <div><p className="font-medium text-foreground">3. Create a rule</p><p className="mt-1 text-xs text-muted">Choose Create alert on a saved list, select a starter condition, and preview current matches before saving.</p></div>
        </div>
      </Panel>

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

      <label className="block max-w-xs text-xs text-muted">Workspace PIN<input type="password" value={pin} onChange={(event) => setPin(event.target.value)} maxLength={128} autoComplete="current-password" className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-foreground" /><span className="mt-1 block text-[12px] text-faint">The server operator provides this PIN. It is required only when saving changes.</span></label>

      {error && <p className="rounded border border-rose-500/50 px-3 py-2 text-sm text-rose-300">{error}</p>}

      <div className="grid gap-4 xl:grid-cols-[22rem_1fr]">
        <div ref={formRef} className="scroll-mt-4">
        <Panel title={editing ? `Edit ${editing.name}` : "New watchlist"}>
          <fieldset disabled={pending || loading} className="space-y-3 disabled:opacity-70">
            {!editing && <div><p className="mb-2 text-xs text-muted">Start with a focused example</p><div className="flex flex-wrap gap-2">{STARTER_LISTS.map((starter) => <button key={starter.name} type="button" onClick={() => applyStarterList(starter)} className="rounded-full border border-border px-2.5 py-1 text-xs text-muted hover:border-accent hover:text-foreground">{starter.name}</button>)}</div></div>}
            <label className="block text-xs text-muted">Name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={64} className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-foreground" /></label>
            <div>
              <div className="mb-2 flex items-center justify-between gap-2 text-xs text-muted">
                <span>Symbols this list will monitor</span>
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
        </div>

        <Panel title="Saved watchlists" right={<span className="font-mono text-xs text-muted">{loading ? "..." : `${data?.items.length ?? 0} lists`}</span>} bodyClassName="p-0">
          <div className="divide-y divide-border">
            {loading && <p className="px-4 py-10 text-center text-sm text-muted">Loading watchlists...</p>}
            {(data?.items ?? []).map((item) => (
              <div key={item.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                <div><div className="flex flex-wrap items-center gap-2"><p className="font-medium text-foreground">{item.name}</p><span className="text-[12px] text-faint">{item.symbols.length} symbols</span></div><div className="mt-1 flex flex-wrap gap-1">{item.symbols.map((symbol) => <span key={symbol} className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-xs text-muted">{symbol}</span>)}</div></div>
                <div className="flex flex-wrap gap-2"><a href={`/alerts?watchlist=${item.id}`} className="rounded border border-accent bg-accent/10 px-2 py-1 text-xs text-foreground hover:bg-accent/20">Create alert</a><button disabled={pending} onClick={() => { setEditing(item); setName(item.name); setSymbols(item.symbols); setSymbolInput(""); formRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40">Edit</button><button disabled={pending || !pin} onClick={() => remove(item)} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative disabled:opacity-40">Delete</button></div>
              </div>
            ))}
            {!loading && !error && data && !data.items.length && <p className="px-4 py-10 text-center text-sm text-muted">No saved watchlists yet. Create one using the form.</p>}
          </div>
        </Panel>
      </div>
    </div>
  );
}
