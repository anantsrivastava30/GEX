"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import BinomialTreeChart from "@/components/charts/BinomialTreeChart";
import Panel from "@/components/ui/Panel";
import { api, BinomialOptionType, BinomialTreeResponse } from "@/lib/api";

const DEFAULT_SYMBOL = "SPY";

function errorMessage(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

function daysUntil(expiration: string) {
  const target = new Date(`${expiration}T00:00:00Z`).getTime();
  const today = new Date();
  const start = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate());
  return Math.max(1, Math.ceil((target - start) / 86_400_000));
}

function decimalInput(value: string) {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed / 100 : Number.NaN;
}

export default function BinomialPage() {
  const expirationController = useRef<AbortController | null>(null);
  const priceController = useRef<AbortController | null>(null);
  const [symbol, setSymbol] = useState(DEFAULT_SYMBOL);
  const [activeSymbol, setActiveSymbol] = useState(DEFAULT_SYMBOL);
  const [expirations, setExpirations] = useState<string[] | null>(null);
  const [expiration, setExpiration] = useState("");
  const [strike, setStrike] = useState("");
  const [optionType, setOptionType] = useState<BinomialOptionType>("call");
  const [steps, setSteps] = useState(8);
  const [daysToExp, setDaysToExp] = useState(30);
  const [rateOverride, setRateOverride] = useState("");
  const [ivOverride, setIvOverride] = useState("");
  const [tree, setTree] = useState<BinomialTreeResponse | null>(null);
  const [loadingExpirations, setLoadingExpirations] = useState(false);
  const [pricing, setPricing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadExpirations(DEFAULT_SYMBOL);
    return () => {
      expirationController.current?.abort();
      priceController.current?.abort();
    };
  }, []);

  function loadExpirations(nextSymbol: string) {
    const normalized = nextSymbol.trim().toUpperCase();
    if (!normalized) {
      setError("Enter a ticker to load expirations.");
      return;
    }
    expirationController.current?.abort();
    const controller = new AbortController();
    expirationController.current = controller;
    setLoadingExpirations(true);
    setError(null);
    setTree(null);
    api.expirations(normalized, controller.signal).then((nextExpirations) => {
      if (expirationController.current !== controller) return;
      const first = nextExpirations[0] ?? "";
      setActiveSymbol(normalized);
      setExpirations(nextExpirations);
      setExpiration(first);
      if (first) setDaysToExp(daysUntil(first));
      return api.tickerSnapshot(normalized, controller.signal);
    }).then((snapshot) => {
      if (snapshot && expirationController.current === controller && snapshot.last != null) {
        setStrike(snapshot.last.toFixed(2));
      }
    }).catch((cause) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (expirationController.current === controller) {
        setExpirations([]);
        setExpiration("");
        setError(errorMessage(cause));
      }
    }).finally(() => {
      if (expirationController.current === controller) setLoadingExpirations(false);
    });
  }

  function selectExpiration(nextExpiration: string) {
    setExpiration(nextExpiration);
    setDaysToExp(daysUntil(nextExpiration));
    setTree(null);
  }

  async function price(event: FormEvent) {
    event.preventDefault();
    const numericStrike = Number(strike);
    const rate = decimalInput(rateOverride);
    const iv = decimalInput(ivOverride);
    if (!expiration) return setError("Choose an expiration before pricing.");
    if (!Number.isFinite(numericStrike) || numericStrike <= 0) return setError("Enter a positive strike.");
    if (!Number.isInteger(daysToExp) || daysToExp < 1) return setError("Pricing-horizon DTE must be at least one day.");
    if (Number.isNaN(rate) || (rate != null && (rate < -25 || rate > 100))) return setError("Rate override must be between -25% and 100%.");
    if (Number.isNaN(iv) || (iv != null && (iv <= 0 || iv > 500))) return setError("IV override must be greater than 0% and at most 500%.");

    priceController.current?.abort();
    const controller = new AbortController();
    priceController.current = controller;
    setPricing(true);
    setError(null);
    setTree(null);
    try {
      const result = await api.binomialTree(activeSymbol, {
        expiration,
        strike: numericStrike,
        optionType,
        steps,
        daysToExp,
        rateOverride: rate,
        ivOverride: iv,
      }, controller.signal);
      if (priceController.current === controller) setTree(result);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError") && priceController.current === controller) setError(errorMessage(cause));
    } finally {
      if (priceController.current === controller) setPricing(false);
    }
  }

  const manualCalibration = rateOverride.trim() || ivOverride.trim();

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-lg font-semibold">Binomial Tree Explorer</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted">European CRR pricing only. The selected pricing horizon is explicit and can differ from the calendar time to expiration.</p>
        </div>
        <span className="rounded border border-border bg-surface px-2 py-1 font-mono text-xs text-muted">2-50 steps</span>
      </div>

      <Panel title="Contract and Horizon">
        <form onSubmit={price} className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <label className="text-xs text-muted">Ticker<input value={symbol} onChange={(event) => setSymbol(event.target.value.toUpperCase())} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent" aria-label="Ticker" /></label>
          <div className="self-end"><button type="button" onClick={() => loadExpirations(symbol)} disabled={loadingExpirations} className="w-full rounded border border-border bg-surface-2 px-3 py-2 text-sm hover:bg-surface-hover disabled:opacity-50">{loadingExpirations ? "Loading expirations..." : "Load expirations"}</button></div>
          <label className="text-xs text-muted">Expiration<select value={expiration} onChange={(event) => selectExpiration(event.target.value)} disabled={!expirations?.length} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent">{!expirations?.length && <option>{loadingExpirations ? "Loading..." : "No expirations"}</option>}{expirations?.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label className="text-xs text-muted">Strike<input type="number" min="0.01" step="0.01" value={strike} onChange={(event) => setStrike(event.target.value)} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent" /></label>
          <label className="text-xs text-muted">Payoff<select value={optionType} onChange={(event) => setOptionType(event.target.value as BinomialOptionType)} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-accent"><option value="call">Call</option><option value="put">Put</option></select></label>
          <label className="text-xs text-muted">Steps: {steps}<input type="range" min="2" max="50" value={steps} onChange={(event) => setSteps(Number(event.target.value))} className="mt-2 w-full accent-accent" /><span className="flex justify-between font-mono text-[10px] text-faint"><span>2</span><span>50</span></span></label>
          <label className="text-xs text-muted">Pricing-horizon DTE<input type="number" min="1" max="3650" value={daysToExp} onChange={(event) => setDaysToExp(Number(event.target.value))} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent" /></label>
          <div className="self-end"><button type="submit" disabled={pricing || loadingExpirations || !expiration} className="w-full rounded bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-strong disabled:opacity-50">{pricing ? "Pricing tree..." : "Price European tree"}</button></div>
        </form>
      </Panel>

      <Panel title="Optional Manual Calibration" right={<span className={`text-xs ${manualCalibration ? "text-warning" : "text-muted"}`}>{manualCalibration ? "Manual input active" : "Endpoint market inputs when available"}</span>}>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="text-xs text-muted">Annual risk-free rate (%)<input type="number" min="-25" max="100" step="0.01" placeholder="Use endpoint market rate" value={rateOverride} onChange={(event) => setRateOverride(event.target.value)} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent" /></label>
          <label className="text-xs text-muted">Annual implied volatility (%)<input type="number" min="0.01" max="500" step="0.01" placeholder="Use selected contract IV" value={ivOverride} onChange={(event) => setIvOverride(event.target.value)} className="mt-1 w-full rounded border border-border bg-background px-3 py-2 font-mono text-sm text-foreground outline-none focus:border-accent" /></label>
        </div>
        <p className="mt-3 text-xs text-muted">Overrides are decimal inputs expressed as percentages here. A manual value is not market calibration. If contract IV is unavailable, supply an IV override to price the tree.</p>
      </Panel>

      {error && <p className="rounded-md border border-negative/50 bg-surface px-4 py-3 text-sm text-negative">Unable to price this tree: {error}</p>}
      {pricing && <div className="h-80 animate-pulse rounded-lg border border-border bg-surface" aria-label="Pricing binomial tree" />}
      {!pricing && !tree && !error && expirations !== null && <p className="rounded-md border border-border bg-surface px-4 py-6 text-center text-sm text-muted">Choose the contract inputs, then price the European tree.</p>}

      {tree && <>
        <div className="grid gap-3 md:grid-cols-4">
          <Metric label="European value" value={`$${tree.nodes.find((node) => node.step === 0)?.option.toFixed(2) ?? "-"}`} />
          <Metric label="Spot" value={`$${tree.spot.toFixed(2)}`} />
          <Metric label="Rate" value={`${(tree.calibration.risk_free_rate * 100).toFixed(3)}%`} source={tree.calibration.risk_free_rate_source} />
          <Metric label="Implied volatility" value={`${(tree.calibration.implied_volatility * 100).toFixed(2)}%`} source={tree.calibration.implied_volatility_source} />
        </div>
        <Panel title={`European ${tree.option_type} CRR tree`} right={<span className="font-mono text-xs text-muted">{tree.symbol} {tree.expiration} | K ${tree.strike.toFixed(2)} | {tree.days_to_exp} DTE</span>}>
          <BinomialTreeChart nodes={tree.nodes} steps={tree.steps} />
        </Panel>
      </>}
    </div>
  );
}

function Metric({ label, value, source }: { label: string; value: string; source?: "market" | "override" }) {
  return <div className="rounded-lg border border-border bg-surface p-3"><p className="text-[10px] font-medium uppercase tracking-wide text-muted">{label}</p><p className="mt-1 font-mono text-lg">{value}</p>{source && <p className={`mt-1 text-xs ${source === "override" ? "text-warning" : "text-muted"}`}>{source === "override" ? "Manual override" : "Endpoint market source"}</p>}</div>;
}
