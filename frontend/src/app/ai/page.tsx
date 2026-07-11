"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  api,
  AIAnalyzeResponse,
  AIHistoryItem,
  AIStatus,
} from "@/lib/api";
import Panel from "@/components/ui/Panel";
import StatTile from "@/components/ui/StatTile";

const DEFAULT_TICKER = "SPY";

function errorMessage(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

function formatTimestamp(value: string | null | undefined) {
  return value ? value.replace("T", " ").replace("Z", "").slice(0, 16) : "-";
}

function formatExpirations(value: unknown) {
  return Array.isArray(value) ? value.join(", ") : "-";
}

export default function AIAnalysisPage() {
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [activeTicker, setActiveTicker] = useState(DEFAULT_TICKER);
  const [expirations, setExpirations] = useState<string[] | null>(null);
  const [selectedExpirations, setSelectedExpirations] = useState<string[]>([]);
  const [offset, setOffset] = useState(35);
  const [reviewing, setReviewing] = useState(false);
  const [pin, setPin] = useState("");
  const [result, setResult] = useState<AIAnalyzeResponse | null>(null);
  const [history, setHistory] = useState<AIHistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingExpirations, setLoadingExpirations] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const expirationController = useRef<AbortController | null>(null);
  const analysisInFlight = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    api.aiStatus(controller.signal).then(setStatus).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError(errorMessage(cause));
      }
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    loadExpirations(DEFAULT_TICKER);
    return () => expirationController.current?.abort();
  }, []);

  function loadExpirations(symbol: string) {
    const normalized = symbol.trim().toUpperCase();
    if (!normalized) {
      setError("Enter a ticker to load expirations.");
      return;
    }
    expirationController.current?.abort();
    const controller = new AbortController();
    expirationController.current = controller;
    setLoadingExpirations(true);
    setError(null);
    setReviewing(false);
    setResult(null);
    api.expirations(normalized, controller.signal).then((nextExpirations) => {
      if (expirationController.current !== controller) return;
      setActiveTicker(normalized);
      setExpirations(nextExpirations);
      setSelectedExpirations(nextExpirations.slice(0, 4));
    }).catch((cause) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (expirationController.current === controller) {
        setExpirations([]);
        setSelectedExpirations([]);
        setError(errorMessage(cause));
      }
    }).finally(() => {
      if (expirationController.current === controller) setLoadingExpirations(false);
    });
  }

  function toggleExpiration(expiration: string) {
    setSelectedExpirations((current) => {
      if (current.includes(expiration)) return current.filter((item) => item !== expiration);
      return current.length < 8 ? [...current, expiration] : current;
    });
  }

  async function loadHistory(historyPin: string) {
    if (!historyPin) {
      setError("Enter the AI PIN to load recent analyses.");
      return;
    }
    setLoadingHistory(true);
    try {
      setHistory(await api.aiHistory(historyPin));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoadingHistory(false);
    }
  }

  async function submitAnalysis() {
    if (analysisInFlight.current || !status) return;
    if (!selectedExpirations.length) {
      setError("Select at least one expiration.");
      return;
    }
    if (status.pin_required && !pin) {
      setError("Enter the AI PIN to run analysis.");
      return;
    }

    analysisInFlight.current = true;
    setSubmitting(true);
    setError(null);
    setResult(null);
    const requestPin = pin;
    try {
      const analysis = await api.aiAnalyze({
        symbol: activeTicker,
        expirations: selectedExpirations,
        model: status.default_model,
        offset,
        pin: requestPin || undefined,
      });
      setResult(analysis);
      if (requestPin) await loadHistory(requestPin);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setPin("");
      setSubmitting(false);
      analysisInFlight.current = false;
    }
  }

  const unavailable = status != null && (!status.openai_configured || !status.pin_required);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-lg font-semibold">AI Analysis</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            Build a positioning snapshot and request a research narrative. This output is
            for research and education only, not financial advice.
          </p>
        </div>
        <span className={`rounded px-2 py-1 text-xs ${unavailable ? "bg-negative/15 text-negative" : "bg-positive/15 text-positive"}`}>
          {status == null ? "Checking service" : unavailable ? "Unavailable" : "Available"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        <StatTile label="OpenAI" value={status == null ? "-" : status.openai_configured ? "Configured" : "Missing"} accent />
        <StatTile label="Access" value={status == null ? "-" : status.pin_required ? "PIN required" : "Disabled"} />
        <StatTile label="Configured model" value={status?.default_model ?? "-"} />
      </div>

      {error && <p role="alert" className="rounded-md border border-negative/40 bg-negative/10 px-4 py-3 text-sm text-foreground">{error}</p>}

      <Panel title="Analysis request" right={<span className="font-mono text-xs text-muted">Up to 8 expirations</span>}>
        <form className="space-y-4" onSubmit={(event: FormEvent) => { event.preventDefault(); loadExpirations(ticker); }}>
          <div className="flex flex-wrap items-end gap-3">
            <label className="grid gap-1.5 text-xs text-muted">
              Ticker
              <input value={ticker} onChange={(event) => setTicker(event.target.value.toUpperCase())} maxLength={12} className="w-36 rounded-md border border-border bg-surface-2 px-3 py-2 font-mono text-sm uppercase text-foreground outline-none focus:border-accent" />
            </label>
            <label className="grid gap-1.5 text-xs text-muted">
              Strike offset
              <input type="number" min="1" max="100" value={offset} onChange={(event) => setOffset(Math.max(1, Math.min(100, Number(event.target.value) || 1)))} className="w-28 rounded-md border border-border bg-surface-2 px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent" />
            </label>
            <button type="submit" disabled={loadingExpirations} className="rounded-md border border-border bg-surface-2 px-4 py-2 text-sm text-foreground hover:bg-surface-hover disabled:cursor-wait disabled:opacity-60">
              {loadingExpirations ? "Loading..." : "Load expirations"}
            </button>
          </div>

          <fieldset disabled={loadingExpirations || !expirations?.length}>
            <legend className="mb-2 text-xs text-muted">Expirations for {activeTicker}</legend>
            <div className="flex max-h-40 flex-wrap gap-2 overflow-y-auto">
              {expirations?.map((expiration) => {
                const checked = selectedExpirations.includes(expiration);
                const atLimit = !checked && selectedExpirations.length >= 8;
                return <label key={expiration} className={`cursor-pointer rounded-md border px-3 py-1.5 font-mono text-xs ${checked ? "border-accent bg-accent/15 text-foreground" : "border-border bg-surface-2 text-muted"} ${atLimit ? "opacity-50" : ""}`}>
                  <input type="checkbox" className="sr-only" checked={checked} disabled={atLimit} onChange={() => toggleExpiration(expiration)} />
                  {expiration}
                </label>;
              })}
            </div>
            {expirations?.length === 0 && <p className="text-sm text-muted">No expirations available for this ticker.</p>}
          </fieldset>

          <button type="button" disabled={!status || unavailable || !selectedExpirations.length} onClick={() => { setError(null); setReviewing(true); }} className="rounded-md border border-accent bg-accent/15 px-4 py-2 text-sm text-foreground hover:bg-accent/25 disabled:cursor-not-allowed disabled:opacity-50">
            Review request
          </button>
        </form>
      </Panel>

      {reviewing && status && <Panel title="Request review" right={<span className="font-mono text-xs text-muted">{status.default_model}</span>}>
        <div className="space-y-4">
          <p className="text-sm text-muted"><span className="font-mono text-foreground">{activeTicker}</span> across {selectedExpirations.length} expiration{selectedExpirations.length === 1 ? "" : "s"}, using a +/- {offset} strike range.</p>
          <label className="grid max-w-sm gap-1.5 text-xs text-muted">
            AI PIN
            <input type="password" autoComplete="off" value={pin} onChange={(event) => setPin(event.target.value)} placeholder="Required for protected requests" className="rounded-md border border-border bg-surface-2 px-3 py-2 text-sm text-foreground outline-none focus:border-accent" />
          </label>
          <div className="flex flex-wrap gap-3">
            <button type="button" disabled={submitting} onClick={submitAnalysis} className="rounded-md border border-accent bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-strong disabled:cursor-wait disabled:opacity-60">
              {submitting ? "Analyzing..." : "Run analysis"}
            </button>
            <button type="button" disabled={submitting} onClick={() => { setReviewing(false); setPin(""); }} className="px-3 py-2 text-sm text-muted hover:text-foreground">Cancel</button>
          </div>
        </div>
      </Panel>}

      {result && <>
        <Panel title={`${result.symbol} analysis`} right={<span className="font-mono text-xs text-muted">{result.prompt_tokens == null ? "" : `${result.prompt_tokens.toLocaleString()} prompt tokens`}</span>}>
          <div className="whitespace-pre-wrap text-sm leading-6 text-foreground">{result.response}</div>
        </Panel>
        <details className="rounded-lg border border-border bg-surface">
          <summary className="cursor-pointer px-4 py-3 text-xs font-medium uppercase tracking-wide text-muted">Analysis payload</summary>
          <pre className="max-h-[32rem] overflow-auto border-t border-border p-4 font-mono text-xs leading-5 text-muted">{JSON.stringify(result.payload, null, 2)}</pre>
        </details>
      </>}

      <Panel title="Recent protected analyses" right={<button type="button" disabled={loadingHistory || !pin} onClick={() => { const historyPin = pin; void loadHistory(historyPin).finally(() => setPin("")); }} className="text-xs text-accent hover:text-accent-strong disabled:cursor-not-allowed disabled:text-faint">{loadingHistory ? "Loading..." : "Load with PIN"}</button>} bodyClassName="p-0">
        {history == null ? <p className="px-4 py-8 text-center text-sm text-muted">Enter the PIN in request review, then load recent analyses.</p> : history.length === 0 ? <p className="px-4 py-8 text-center text-sm text-muted">No saved analyses yet.</p> : <div className="max-h-[30rem] overflow-auto"><table className="w-full min-w-[680px] text-sm"><thead className="sticky top-0 bg-surface-2 text-left text-xs text-muted"><tr><th className="px-4 py-2 font-medium">Time</th><th className="px-4 py-2 font-medium">Ticker</th><th className="px-4 py-2 font-medium">Expirations</th><th className="px-4 py-2 text-right font-medium">Tokens</th><th className="px-4 py-2 font-medium">Narrative</th></tr></thead><tbody>{history.map((item, index) => <tr key={`${item.ts}-${item.ticker}-${index}`} className="border-t border-border align-top hover:bg-surface-hover"><td className="whitespace-nowrap px-4 py-2 font-mono text-xs text-muted">{formatTimestamp(item.ts)}</td><td className="px-4 py-2 font-mono">{item.ticker ?? "-"}</td><td className="px-4 py-2 font-mono text-xs text-muted">{formatExpirations(item.expirations)}</td><td className="px-4 py-2 text-right font-mono text-xs">{String(item.token_count ?? "-")}</td><td className="max-w-md px-4 py-2 text-xs leading-5 text-muted">{item.response ?? "-"}</td></tr>)}</tbody></table></div>}
      </Panel>
    </div>
  );
}
