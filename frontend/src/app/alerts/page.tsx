"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import FilterBuilder from "@/components/screener/FilterBuilder";
import Panel from "@/components/ui/Panel";
import {
  AlertEventsResponse,
  AlertRule,
  AlertStatus,
  api,
  CustomScreenerRequest,
  CustomScreenerResponse,
  ScreenerCondition,
  ScreenerFieldSpec,
  WatchlistListResponse,
} from "@/lib/api";

const DEFAULT_QUERY: CustomScreenerRequest = {
  scope: "contract",
  conditions: [
    { field: "volume_oi", operator: "gte", value: 2 },
    { field: "open_interest", operator: "gte", value: 100 },
  ],
  sort: { field: "volume_oi", direction: "desc" },
  limit: 25,
};

const ALERT_TEMPLATES: { name: string; description: string; query: CustomScreenerRequest }[] = [
  {
    name: "High contract activity",
    description: "Contracts trading at least 2x current OI, with OI of 100 or more.",
    query: DEFAULT_QUERY,
  },
  {
    name: "Strong gamma magnet",
    description: "Tickers in positive gamma with a gamma-gap score of at least 25.",
    query: {
      scope: "ticker",
      conditions: [
        { field: "gamma_positive_zone", operator: "eq", value: true },
        { field: "gamma_gap_score", operator: "gte", value: 25 },
      ],
      sort: { field: "gamma_gap_score", direction: "desc" },
      limit: 25,
    },
  },
  {
    name: "Put-heavy activity",
    description: "Tickers where put volume is at least 1.25x call volume.",
    query: {
      scope: "ticker",
      conditions: [{ field: "pc_volume_ratio", operator: "gte", value: 1.25 }],
      sort: { field: "pc_volume_ratio", direction: "desc" },
      limit: 25,
    },
  },
  {
    name: "Crossed above 50-day SMA",
    description: "Price moved from at or below its 50-day average to above it today.",
    query: {
      scope: "ticker",
      conditions: [{ field: "sma_50_cross", operator: "eq", value: "above" }],
      sort: { field: "price_vs_sma_50_pct", direction: "desc" },
      limit: 50,
    },
  },
  {
    name: "Crossed below 50-day SMA",
    description: "Price moved from at or above its 50-day average to below it today.",
    query: {
      scope: "ticker",
      conditions: [{ field: "sma_50_cross", operator: "eq", value: "below" }],
      sort: { field: "price_vs_sma_50_pct", direction: "asc" },
      limit: 50,
    },
  },
  {
    name: "Crossed above 200-day SMA",
    description: "Price moved from at or below its 200-day average to above it today.",
    query: {
      scope: "ticker",
      conditions: [{ field: "sma_200_cross", operator: "eq", value: "above" }],
      sort: { field: "price_vs_sma_200_pct", direction: "desc" },
      limit: 50,
    },
  },
  {
    name: "Crossed below 200-day SMA",
    description: "Price moved from at or above its 200-day average to below it today.",
    query: {
      scope: "ticker",
      conditions: [{ field: "sma_200_cross", operator: "eq", value: "below" }],
      sort: { field: "price_vs_sma_200_pct", direction: "asc" },
      limit: 50,
    },
  },
];

const OPERATOR_LABELS: Record<ScreenerCondition["operator"], string> = {
  eq: "=", ne: "!=", gt: ">", gte: ">=", lt: "<", lte: "<=",
};

function conditionHasValue(condition: ScreenerCondition) {
  return typeof condition.value !== "string" || condition.value.trim() !== "";
}

function validQuery(query: CustomScreenerRequest) {
  return query.conditions.length > 0 && query.conditions.every(conditionHasValue);
}

function copyQuery(query: CustomScreenerRequest): CustomScreenerRequest {
  return {
    ...query,
    conditions: query.conditions.map((condition) => ({ ...condition })),
    sort: query.sort ? { ...query.sort } : undefined,
  };
}

function describeQuery(query: CustomScreenerRequest, fields: ScreenerFieldSpec[]) {
  if (!query.conditions.length) return "No conditions configured";
  return query.conditions.map((condition) => {
    const label = fields.find((field) => field.name === condition.field)?.label ?? condition.field.replaceAll("_", " ");
    const value = conditionHasValue(condition) ? String(condition.value) : "(empty)";
    return `${label} ${OPERATOR_LABELS[condition.operator]} ${value}`;
  }).join(" AND ");
}

export default function AlertsPage() {
  const eventSequence = useRef(0);
  const metadataSequence = useRef(0);
  const formRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<AlertStatus | null>(null);
  const [watchlists, setWatchlists] = useState<WatchlistListResponse | null>(null);
  const [rules, setRules] = useState<AlertRule[] | null>(null);
  const [events, setEvents] = useState<AlertEventsResponse | null>(null);
  const [preview, setPreview] = useState<CustomScreenerResponse | null>(null);
  const [fields, setFields] = useState<ScreenerFieldSpec[]>([]);
  const [editing, setEditing] = useState<AlertRule | null>(null);
  const [name, setName] = useState("");
  const [watchlistId, setWatchlistId] = useState<number | "all">("all");
  const [query, setQuery] = useState<CustomScreenerRequest>(DEFAULT_QUERY);
  const [enabled, setEnabled] = useState(true);
  const [discord, setDiscord] = useState(false);
  const [email, setEmail] = useState(false);
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [pending, startTransition] = useTransition();

  async function load(signal?: AbortSignal) {
    const sequence = ++metadataSequence.current;
    try {
      const [nextStatus, nextWatchlists, nextRules, nextFields] = await Promise.all([
        api.alertStatus(signal), api.watchlists(signal), api.alertRules(signal), api.screenerFields(signal),
      ]);
      if (metadataSequence.current === sequence) {
        setStatus(nextStatus); setWatchlists(nextWatchlists); setRules(nextRules); setFields(nextFields.fields);
        const requestedId = typeof window === "undefined" ? 0 : Number(new URLSearchParams(window.location.search).get("watchlist"));
        const requestedExists = nextWatchlists.items.some((item) => item.id === requestedId);
        if (requestedExists) setWatchlistId(requestedId);
      }
    } finally {
      if (metadataSequence.current === sequence) setLoading(false);
    }
  }

  function refreshEvents(filter = unreadOnly, signal?: AbortSignal) {
    const sequence = ++eventSequence.current;
    return api.alertEvents(filter, 100, signal).then((response) => {
      if (eventSequence.current === sequence) setEvents(response);
    }).finally(() => {
      if (eventSequence.current === sequence) setEventsLoading(false);
    });
  }

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) setError(cause instanceof Error ? cause.message : String(cause));
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const refresh = (signal?: AbortSignal) => {
      return refreshEvents(unreadOnly, signal);
    };
    refresh(controller.signal).catch((cause) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) setError(cause instanceof Error ? cause.message : String(cause));
    });
    const timer = window.setInterval(() => refresh().catch(() => undefined), 60_000);
    return () => { controller.abort(); window.clearInterval(timer); };
    // Event polling is independent from workspace metadata.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unreadOnly]);

  function resetForm() {
    setEditing(null); setName(""); setQuery(copyQuery(DEFAULT_QUERY)); setEnabled(true); setDiscord(false); setEmail(false); setPreview(null);
    setWatchlistId("all");
  }

  function save() {
    setError(null);
    if (!validQuery(query)) {
      setError("Every alert condition requires a value.");
      return;
    }
    startTransition(async () => {
      try {
        const request = { name, enabled, watchlist_id: watchlistId === "all" ? null : watchlistId, query, notify_discord: discord, notify_email: email };
        if (editing) await api.updateAlertRule(editing.id, request, pin); else await api.createAlertRule(request, pin);
        setLoading(true);
        await load(); await refreshEvents(); resetForm();
      } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    });
  }

  function edit(rule: AlertRule) {
    setEditing(rule); setName(rule.name); setWatchlistId(rule.watchlist_id ?? "all"); setQuery(copyQuery(rule.query)); setEnabled(rule.enabled); setDiscord(rule.notify_discord); setEmail(rule.notify_email); setPreview(null);
    formRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function applyTemplate(template: (typeof ALERT_TEMPLATES)[number]) {
    setQuery(copyQuery(template.query));
    if (!editing) setName(template.name);
    setPreview(null);
    setError(null);
  }

  function previewMatches() {
    const watchlist = watchlistId === "all" ? null : watchlists?.items.find((item) => item.id === watchlistId);
    if ((watchlistId !== "all" && !watchlist) || !validQuery(query)) {
      setError("Choose a symbol scope and complete every condition before previewing.");
      return;
    }
    setPreviewLoading(true);
    setPreview(null);
    setError(null);
    api.customScreener({
      ...copyQuery(query),
      symbols: watchlist?.symbols,
      limit: query.scope === "ticker" ? watchlists?.snapshot_symbol_limit ?? 50 : 25,
    }).then(setPreview).catch((cause) => {
      setError(cause instanceof Error ? cause.message : String(cause));
    }).finally(() => setPreviewLoading(false));
  }

  function remove(rule: AlertRule) {
    if (!window.confirm(`Delete ${rule.name}?`)) return;
    startTransition(async () => { try { await api.deleteAlertRule(rule.id, pin); setLoading(true); await load(); if (editing?.id === rule.id) resetForm(); } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); } });
  }

  function markAllRead() {
    setError(null);
    startTransition(async () => {
      try { await api.markAllAlertsRead(pin); await refreshEvents(); }
      catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    });
  }

  function markRead(eventId: number) {
    setError(null);
    startTransition(async () => {
      try { await api.markAlertRead(eventId, pin); await refreshEvents(); }
      catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    });
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Alerts</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">Apply a rule to a focused watchlist or every scheduled symbol, preview it against current snapshots, and retain each match in the in-app inbox.</p>
      </div>

      <Panel title="What actually triggers an alert">
        <div className="grid gap-3 text-sm md:grid-cols-4">
          <div><p className="font-medium text-foreground">1. Symbol scope</p><p className="mt-1 text-xs text-muted">Choose all scheduled symbols or one focused watchlist. A watchlist alone never sends an alert.</p></div>
          <div><p className="font-medium text-foreground">2. Snapshot</p><p className="mt-1 text-xs text-muted">The server captures option snapshots and maintains daily closes for SMA rules.</p></div>
          <div><p className="font-medium text-foreground">3. Rule match</p><p className="mt-1 text-xs text-muted">Every condition must match when rules evaluate about every 30 minutes.</p></div>
          <div><p className="font-medium text-foreground">4. Delivery</p><p className="mt-1 text-xs text-muted">In-app is always on. Discord and email work only when configured below.</p></div>
        </div>
        <p className="mt-3 border-t border-border pt-3 text-xs text-faint">Rules evaluate snapshots, not every market tick. SMA crossings compare the prior completed Tradier close with the latest snapshot spot and emit once per rule version and market session.</p>
      </Panel>

      <p className="rounded border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">Shared server workspace. Every connected browser can view the same rules and inbox. Changes require the operator-provided workspace PIN.</p>

      <label className="block max-w-xs text-xs text-muted">Workspace PIN<input type="password" value={pin} onChange={(event) => setPin(event.target.value)} maxLength={128} autoComplete="current-password" className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-foreground" /><span className="mt-1 block text-[12px] text-faint">Required only to save rules or manage inbox state.</span></label>

      <div className="flex flex-wrap gap-2 text-xs">
        <span className={`rounded border px-2 py-1 ${status ? status.scheduler_enabled ? "border-emerald-500/50 text-emerald-300" : "border-rose-500/50 text-rose-300" : "border-border text-muted"}`}>Scheduler {status ? status.scheduler_enabled ? "enabled" : "disabled" : "checking..."}</span>
        <span className={`rounded border px-2 py-1 ${watchlists?.tradier_configured ? "border-emerald-500/50 text-emerald-300" : watchlists ? "border-rose-500/50 text-rose-300" : "border-border text-muted"}`}>Market data {watchlists ? watchlists.tradier_configured ? "connected" : "not configured" : "checking..."}</span>
        <span className="rounded border border-emerald-500/50 px-2 py-1 text-emerald-300">In-app on</span>
        <span className={`rounded border px-2 py-1 ${status?.discord_configured ? "border-emerald-500/50 text-emerald-300" : "border-border text-muted"}`}>Discord {status ? status.discord_configured ? "configured" : "off" : "checking..."}</span>
        <span className={`rounded border px-2 py-1 ${status?.email_configured ? "border-emerald-500/50 text-emerald-300" : "border-border text-muted"}`}>Email {status ? status.email_configured ? "configured" : "off" : "checking..."}</span>
        <span className="rounded border border-border px-2 py-1 text-muted">{eventsLoading ? "... unread" : `${events?.unread ?? 0} unread`}</span>
      </div>
      {status && !status.scheduler_enabled && <p className="rounded border border-rose-500/40 bg-rose-500/5 px-3 py-2 text-xs text-rose-200">Alert automation is blocked because the scheduler is disabled.</p>}
      {watchlists && !watchlists.tradier_configured && <p className="rounded border border-rose-500/40 bg-rose-500/5 px-3 py-2 text-xs text-rose-200">Alert automation is blocked because Tradier market data is not configured.</p>}
      {error && <p className="rounded border border-rose-500/50 px-3 py-2 text-sm text-rose-300">{error}</p>}

      <div ref={formRef} className="scroll-mt-4">
      <Panel title={editing ? `Edit ${editing.name}` : "New alert rule"}>
        <fieldset disabled={pending || loading}>
        <div>
          <p className="text-xs text-muted">Start with a practical structure, then adjust its thresholds.</p>
          <div className="mt-2 grid gap-2 md:grid-cols-3 xl:grid-cols-4">
            {ALERT_TEMPLATES.map((template) => <button key={template.name} type="button" onClick={() => applyTemplate(template)} className="rounded border border-border bg-surface-2 px-3 py-2 text-left hover:border-accent"><span className="block text-sm font-medium text-foreground">{template.name}</span><span className="mt-1 block text-xs text-muted">{template.description}</span></button>)}
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="mt-4 text-xs text-muted">Rule name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={64} placeholder="Example: Strong semiconductor magnets" className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm" /></label>
          <label className="mt-4 text-xs text-muted">Symbol scope<select value={String(watchlistId)} onChange={(event) => { setWatchlistId(event.target.value === "all" ? "all" : Number(event.target.value)); setPreview(null); }} className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm"><option value="all">All scheduled symbols ({watchlists?.available_symbols.length ?? 0})</option>{(watchlists?.items ?? []).map((item) => <option key={item.id} value={item.id}>{item.name} ({item.symbols.length})</option>)}</select><span className="mt-1 block text-[12px] text-faint">All scheduled is the baseline plus symbols from every shared watchlist and updates automatically.</span></label>
        </div>
        {!loading && watchlists && !watchlists.items.length && <p className="mt-4 rounded border border-border px-3 py-2 text-sm text-muted">All scheduled symbols is ready. Create a <a href="/watchlists" className="underline">focused watchlist</a> only when you want a narrower alert audience.</p>}
        <div className="mt-4">{loading ? <p className="text-sm text-muted">Loading alert fields...</p> : <FilterBuilder fields={fields} value={query} onChange={(next) => { setQuery(next); setPreview(null); }} />}</div>
        <p className={`mt-3 rounded border px-3 py-2 text-xs ${validQuery(query) ? "border-border bg-surface-2 text-muted" : "border-rose-500/40 text-rose-200"}`}><span className="font-medium text-foreground">Rule:</span> Check {query.scope}s where {describeQuery(query, fields)}.</p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button type="button" onClick={previewMatches} disabled={previewLoading || !validQuery(query)} className="rounded border border-border px-3 py-1.5 text-sm text-muted hover:border-accent hover:text-foreground disabled:opacity-40">{previewLoading ? "Checking snapshots..." : "Preview current matches"}</button>
          <span className="text-xs text-faint">Previewing does not create inbox events.</span>
        </div>
        {preview && <div className={`mt-3 rounded border px-3 py-2 text-sm ${preview.stale ? "border-amber-500/40 bg-amber-500/5" : "border-emerald-500/40 bg-emerald-500/5"}`}><div className="flex flex-wrap items-center gap-2"><span className="font-medium text-foreground">{preview.rows.length} current match{preview.rows.length === 1 ? "" : "es"}</span><span className="font-mono text-xs text-muted">Snapshot {preview.as_of ?? "unavailable"}</span></div>{preview.unavailable_symbols.length > 0 && <p className="mt-1 text-xs text-amber-200">Missing snapshots: {preview.unavailable_symbols.join(", ")}. Alerts will wait until the whole selected scope is current.</p>}{preview.technical_unavailable_symbols.length > 0 && <p className="mt-1 text-xs text-amber-200">SMA history is not ready for: {preview.technical_unavailable_symbols.join(", ")}. Other symbols can still match.</p>}{!preview.rows.length && !preview.stale && <p className="mt-1 text-xs text-muted">The rule is valid, but nothing meets it in the latest snapshot.</p>}{preview.rows.length > 0 && <p className="mt-1 text-xs text-muted">First matches: {preview.rows.slice(0, 8).map((row) => row.symbol).join(", ")}</p>}</div>}
        <div className="mt-4 flex flex-wrap gap-4 text-sm"><label><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} className="mr-2" />Enabled</label><label className={!status?.discord_configured ? "text-faint" : ""}><input type="checkbox" checked={discord} disabled={!status?.discord_configured} onChange={(event) => setDiscord(event.target.checked)} className="mr-2" />Discord</label><label className={!status?.email_configured ? "text-faint" : ""}><input type="checkbox" checked={email} disabled={!status?.email_configured} onChange={(event) => setEmail(event.target.checked)} className="mr-2" />Email</label></div>
        <p className="mt-2 text-xs text-faint">In-app events require no channel selection. Discord and email are unavailable until the server operator configures them.</p>
        <div className="mt-4 flex gap-2"><button type="button" onClick={save} disabled={!pin || !name.trim() || !validQuery(query)} className="rounded border border-accent bg-accent/15 px-3 py-1.5 text-sm disabled:opacity-40">{pending ? "Saving..." : editing ? "Update rule" : "Create rule"}</button>{editing && <button type="button" onClick={resetForm} className="rounded border border-border px-3 py-1.5 text-sm text-muted">Cancel</button>}</div>
        </fieldset>
      </Panel>
      </div>

      <Panel title="Rules" right={<span className="font-mono text-xs text-muted">{loading ? "..." : `${rules?.length ?? 0} rules`}</span>} bodyClassName="p-0">
        <div className="divide-y divide-border">
          {loading && <p className="px-4 py-10 text-center text-sm text-muted">Loading alert rules...</p>}
          {(rules ?? []).map((rule) => {
            const isValid = validQuery(rule.query);
            const state = !rule.enabled ? "PAUSED" : !isValid ? "NEEDS SETUP" : rule.last_error ? "BLOCKED" : "ENABLED";
            const stateClass = state === "ENABLED" ? "bg-positive/10 text-positive" : state === "PAUSED" ? "bg-border text-faint" : "bg-rose-500/10 text-rose-300";
            const channels = ["In-app", rule.notify_discord ? "Discord" : null, rule.notify_email ? "Email" : null].filter(Boolean).join(" + ");
            return <div key={rule.id} className="flex flex-wrap items-start justify-between gap-3 px-4 py-3"><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="font-medium">{rule.name}</span><span className={`rounded px-1.5 py-0.5 text-[12px] md:text-[10px] ${stateClass}`}>{state}</span></div><p className="mt-1 text-xs text-muted">{rule.watchlist_name} · {rule.query.scope} scope · {channels}</p><p className="mt-1 break-words text-xs text-faint">{describeQuery(rule.query, fields)}</p>{!isValid && <p className="mt-1 text-xs text-rose-300">This rule cannot match because at least one condition is empty. Edit it and choose a starter rule or enter a value.</p>}{rule.last_error && <p className="mt-1 text-xs text-rose-300">Last evaluation: {rule.last_error}</p>}<p className="mt-1 text-[12px] text-faint">{rule.last_evaluated_at ? `Checked ${rule.last_evaluated_at}` : "Not evaluated yet"}</p></div><div className="flex gap-2"><button disabled={pending} onClick={() => edit(rule)} className="rounded border border-border px-2 py-1 text-xs text-muted disabled:opacity-40">Edit</button><button disabled={pending || !pin} onClick={() => remove(rule)} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative disabled:opacity-40">Delete</button></div></div>;
          })}
          {!loading && !error && rules && !rules.length && <p className="px-4 py-10 text-center text-sm text-muted">No alert rules yet. Choose a starter rule above, preview it, then save.</p>}
        </div>
      </Panel>

      <Panel title="Event inbox" right={<div className="flex items-center gap-2"><label className="text-xs text-muted"><input type="checkbox" checked={unreadOnly} onChange={(event) => { setEvents(null); setEventsLoading(true); setUnreadOnly(event.target.checked); }} className="mr-1" />Unread only</label><button disabled={pending || !pin} onClick={markAllRead} className="rounded border border-border px-2 py-1 text-xs text-muted disabled:opacity-40">Mark all read</button></div>} bodyClassName="p-0">
        <div className="max-h-[34rem] divide-y divide-border overflow-auto">{eventsLoading && <p role="status" className="px-4 py-10 text-center text-sm text-muted">Loading alert events...</p>}{!eventsLoading && (events?.items ?? []).map((event) => <button key={event.id} disabled={pending || !!event.read_at || !pin} onClick={() => markRead(event.id)} className={`block w-full px-4 py-3 text-left hover:bg-surface-hover disabled:cursor-default ${event.read_at ? "opacity-60" : ""}`}><div className="flex justify-between gap-3"><span className="text-sm font-medium">{event.title}</span><span className="whitespace-nowrap font-mono text-xs text-faint">{event.snapshot_date}</span></div><p className="mt-1 text-xs text-muted">{event.message}</p><p className="mt-1 text-[12px] md:text-[10px] text-faint">{event.rule_name} · Discord {event.discord_status} · Email {event.email_status}</p></button>)}{!eventsLoading && !error && events && !events.items.length && <p className="px-4 py-8 text-center text-sm text-muted">No events yet. The inbox fills only after an enabled, valid rule matches a current snapshot. Use Preview current matches to verify a rule before waiting for the next evaluation.</p>}</div>
      </Panel>
    </div>
  );
}
