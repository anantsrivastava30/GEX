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
  ScreenerFieldSpec,
  WatchlistListResponse,
} from "@/lib/api";

const DEFAULT_QUERY: CustomScreenerRequest = {
  scope: "contract",
  conditions: [{ field: "volume_oi", operator: "gte", value: 3 }],
  sort: { field: "volume_oi", direction: "desc" },
  limit: 25,
};

export default function AlertsPage() {
  const eventSequence = useRef(0);
  const metadataSequence = useRef(0);
  const [status, setStatus] = useState<AlertStatus | null>(null);
  const [watchlists, setWatchlists] = useState<WatchlistListResponse | null>(null);
  const [rules, setRules] = useState<AlertRule[] | null>(null);
  const [events, setEvents] = useState<AlertEventsResponse | null>(null);
  const [fields, setFields] = useState<ScreenerFieldSpec[]>([]);
  const [editing, setEditing] = useState<AlertRule | null>(null);
  const [name, setName] = useState("");
  const [watchlistId, setWatchlistId] = useState(0);
  const [query, setQuery] = useState<CustomScreenerRequest>(DEFAULT_QUERY);
  const [enabled, setEnabled] = useState(true);
  const [discord, setDiscord] = useState(false);
  const [email, setEmail] = useState(false);
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [pending, startTransition] = useTransition();

  async function load(signal?: AbortSignal) {
    const sequence = ++metadataSequence.current;
    try {
      const [nextStatus, nextWatchlists, nextRules, nextFields] = await Promise.all([
        api.alertStatus(signal), api.watchlists(signal), api.alertRules(signal), api.screenerFields(signal),
      ]);
      if (metadataSequence.current === sequence) {
        setStatus(nextStatus); setWatchlists(nextWatchlists); setRules(nextRules); setFields(nextFields.fields);
        setWatchlistId((current) => current || nextWatchlists.items[0]?.id || 0);
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
    setEditing(null); setName(""); setQuery(DEFAULT_QUERY); setEnabled(true); setDiscord(false); setEmail(false);
    if (watchlists?.items.length) setWatchlistId(watchlists.items[0].id);
  }

  function save() {
    setError(null);
    startTransition(async () => {
      try {
        const request = { name, enabled, watchlist_id: watchlistId, query, notify_discord: discord, notify_email: email };
        if (editing) await api.updateAlertRule(editing.id, request, pin); else await api.createAlertRule(request, pin);
        setLoading(true);
        await load(); await refreshEvents(); resetForm();
      } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    });
  }

  function edit(rule: AlertRule) {
    setEditing(rule); setName(rule.name); setWatchlistId(rule.watchlist_id); setQuery(rule.query); setEnabled(rule.enabled); setDiscord(rule.notify_discord); setEmail(rule.notify_email);
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
      <div><h1 className="text-lg font-semibold">Alerts</h1><p className="mt-1 max-w-3xl text-sm text-muted">Evaluate custom rules after scheduled snapshots and retain an in-app event trail. Optional delivery uses server-owned Discord and SMTP settings.</p></div>
      <p className="rounded border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-200">Shared server workspace. {status ? `Rules observe persisted snapshots every ${status.evaluation_interval_minutes} minutes.` : "Loading scheduler configuration..."} Mutations require the workspace PIN and are not live price-crossing alerts.</p>

      <label className="block max-w-xs text-xs text-muted">Workspace PIN<input type="password" value={pin} onChange={(event) => setPin(event.target.value)} maxLength={128} autoComplete="current-password" className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm text-foreground" /></label>

      <div className="flex flex-wrap gap-2 text-xs">
        <span className={`rounded border px-2 py-1 ${status ? status.scheduler_enabled ? "border-emerald-500/50 text-emerald-300" : "border-rose-500/50 text-rose-300" : "border-border text-muted"}`}>Scheduler {status ? status.scheduler_enabled ? "enabled" : "disabled" : "checking..."}</span>
        <span className={`rounded border px-2 py-1 ${status?.discord_configured ? "border-emerald-500/50 text-emerald-300" : "border-border text-muted"}`}>Discord {status ? status.discord_configured ? "configured" : "off" : "checking..."}</span>
        <span className={`rounded border px-2 py-1 ${status?.email_configured ? "border-emerald-500/50 text-emerald-300" : "border-border text-muted"}`}>Email {status ? status.email_configured ? "configured" : "off" : "checking..."}</span>
        <span className="rounded border border-border px-2 py-1 text-muted">{eventsLoading ? "... unread" : `${events?.unread ?? 0} unread`}</span>
      </div>
      {error && <p className="rounded border border-rose-500/50 px-3 py-2 text-sm text-rose-300">{error}</p>}

      <Panel title={editing ? `Edit ${editing.name}` : "New alert rule"}>
        <fieldset disabled={pending || loading}>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="text-xs text-muted">Name<input value={name} onChange={(event) => setName(event.target.value)} maxLength={64} className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm" /></label>
          <label className="text-xs text-muted">Watchlist<select value={watchlistId} onChange={(event) => setWatchlistId(Number(event.target.value))} className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm">{(watchlists?.items ?? []).map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        </div>
        {!loading && watchlists && !watchlists.items.length && <p className="mt-4 rounded border border-amber-500/40 px-3 py-2 text-sm text-amber-200">Create a watchlist before creating an alert rule.</p>}
        <div className="mt-4">{loading ? <p className="text-sm text-muted">Loading alert fields...</p> : <FilterBuilder fields={fields} value={query} onChange={setQuery} />}</div>
        <div className="mt-4 flex flex-wrap gap-4 text-sm"><label><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} className="mr-2" />Enabled</label><label className={!status?.discord_configured ? "text-faint" : ""}><input type="checkbox" checked={discord} disabled={!status?.discord_configured} onChange={(event) => setDiscord(event.target.checked)} className="mr-2" />Discord</label><label className={!status?.email_configured ? "text-faint" : ""}><input type="checkbox" checked={email} disabled={!status?.email_configured} onChange={(event) => setEmail(event.target.checked)} className="mr-2" />Email</label></div>
        <div className="mt-4 flex gap-2"><button type="button" onClick={save} disabled={!pin || !name.trim() || !watchlistId || !query.conditions.length} className="rounded border border-accent bg-accent/15 px-3 py-1.5 text-sm disabled:opacity-40">{pending ? "Saving..." : editing ? "Update rule" : "Create rule"}</button>{editing && <button type="button" onClick={resetForm} className="rounded border border-border px-3 py-1.5 text-sm text-muted">Cancel</button>}</div>
        </fieldset>
      </Panel>

      <Panel title="Rules" right={<span className="font-mono text-xs text-muted">{loading ? "..." : `${rules?.length ?? 0} rules`}</span>} bodyClassName="p-0">
        <div className="divide-y divide-border">{loading && <p className="px-4 py-10 text-center text-sm text-muted">Loading alert rules...</p>}{(rules ?? []).map((rule) => <div key={rule.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"><div><div className="flex items-center gap-2"><span className="font-medium">{rule.name}</span><span className={`rounded px-1.5 py-0.5 text-[10px] ${rule.enabled ? "bg-positive/10 text-positive" : "bg-border text-faint"}`}>{rule.enabled ? "ACTIVE" : "PAUSED"}</span></div><p className="mt-1 text-xs text-muted">{rule.watchlist_name} · {rule.query.scope} · {rule.query.conditions.length} condition(s){rule.last_error ? ` · ${rule.last_error}` : ""}</p></div><div className="flex gap-2"><button disabled={pending} onClick={() => edit(rule)} className="rounded border border-border px-2 py-1 text-xs text-muted disabled:opacity-40">Edit</button><button disabled={pending || !pin} onClick={() => remove(rule)} className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative disabled:opacity-40">Delete</button></div></div>)}{!loading && !error && rules && !rules.length && <p className="px-4 py-10 text-center text-sm text-muted">No alert rules yet. Create one using the form above.</p>}</div>
      </Panel>

      <Panel title="Event inbox" right={<div className="flex items-center gap-2"><label className="text-xs text-muted"><input type="checkbox" checked={unreadOnly} onChange={(event) => { setEvents(null); setEventsLoading(true); setUnreadOnly(event.target.checked); }} className="mr-1" />Unread only</label><button disabled={pending || !pin} onClick={markAllRead} className="rounded border border-border px-2 py-1 text-xs text-muted disabled:opacity-40">Mark all read</button></div>} bodyClassName="p-0">
        <div className="max-h-[34rem] divide-y divide-border overflow-auto">{eventsLoading && <p role="status" className="px-4 py-10 text-center text-sm text-muted">Loading alert events...</p>}{!eventsLoading && (events?.items ?? []).map((event) => <button key={event.id} disabled={pending || !!event.read_at || !pin} onClick={() => markRead(event.id)} className={`block w-full px-4 py-3 text-left hover:bg-surface-hover disabled:cursor-default ${event.read_at ? "opacity-60" : ""}`}><div className="flex justify-between gap-3"><span className="text-sm font-medium">{event.title}</span><span className="whitespace-nowrap font-mono text-xs text-faint">{event.snapshot_date}</span></div><p className="mt-1 text-xs text-muted">{event.message}</p><p className="mt-1 text-[10px] text-faint">{event.rule_name} · Discord {event.discord_status} · Email {event.email_status}</p></button>)}{!eventsLoading && !error && events && !events.items.length && <p className="px-4 py-8 text-center text-sm text-muted">No alert events yet. Events appear after a current snapshot matches an enabled rule.</p>}</div>
      </Panel>
    </div>
  );
}
