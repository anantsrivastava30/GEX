"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import AlertInbox from "@/components/alerts/AlertInbox";
import AlertRulesPanel from "@/components/alerts/AlertRulesPanel";
import Tabs from "@/components/ui/Tabs";
import {
  AlertEventsResponse,
  AlertRule,
  AlertRuleMutation,
  AlertStatus,
  api,
  ScreenerFieldSpec,
  WatchlistListResponse,
} from "@/lib/api";

// Alerts workspace. Two surfaces: the inbox you triage, and the rules that
// fill it. The inbox owns the page - rules sit behind a tab so the reading
// surface is not buried under a form.

function Chip({
  label,
  tone = "muted",
}: {
  label: string;
  tone?: "muted" | "good" | "bad";
}) {
  const border =
    tone === "good"
      ? "border-positive/40 text-positive"
      : tone === "bad"
        ? "border-negative/40 text-negative"
        : "border-border text-muted";
  return (
    <span className={`rounded border px-2 py-1 text-xs ${border}`}>{label}</span>
  );
}

export default function AlertsPage() {
  const eventSequence = useRef(0);
  const metadataSequence = useRef(0);
  const [tab, setTab] = useState("inbox");
  const [status, setStatus] = useState<AlertStatus | null>(null);
  const [watchlists, setWatchlists] = useState<WatchlistListResponse | null>(null);
  const [rules, setRules] = useState<AlertRule[] | null>(null);
  const [events, setEvents] = useState<AlertEventsResponse | null>(null);
  const [fields, setFields] = useState<ScreenerFieldSpec[]>([]);
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pending, startTransition] = useTransition();

  async function load(signal?: AbortSignal) {
    const sequence = ++metadataSequence.current;
    try {
      const [nextStatus, nextWatchlists, nextRules, nextFields] = await Promise.all([
        api.alertStatus(signal),
        api.watchlists(signal),
        api.alertRules(signal),
        api.screenerFields(signal),
      ]);
      if (metadataSequence.current === sequence) {
        setStatus(nextStatus);
        setWatchlists(nextWatchlists);
        setRules(nextRules);
        setFields(nextFields.fields);
      }
    } finally {
      if (metadataSequence.current === sequence) setLoading(false);
    }
  }

  function refreshEvents(signal?: AbortSignal) {
    const sequence = ++eventSequence.current;
    return api
      .alertEvents(false, 300, signal)
      .then((response) => {
        if (eventSequence.current === sequence) setEvents(response);
      })
      .finally(() => {
        if (eventSequence.current === sequence) setEventsLoading(false);
      });
  }

  function report(cause: unknown) {
    if (cause instanceof DOMException && cause.name === "AbortError") return;
    setError(cause instanceof Error ? cause.message : String(cause));
  }

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal).catch(report);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    refreshEvents(controller.signal).catch(report);
    const timer = window.setInterval(
      () => refreshEvents().catch(() => undefined),
      60_000,
    );
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  // Triage feels instant: flip the events locally, then reconcile with the
  // server and roll back the optimistic write when the call fails.
  function markRead(ids: number[]) {
    if (!pin || !ids.length) return;
    setError(null);
    const targets = new Set(ids);
    const snapshot = events;
    const stamp = new Date().toISOString();
    setEvents((current) => {
      if (!current) return current;
      const flipped = current.items.filter(
        (event) => targets.has(event.id) && !event.read_at,
      ).length;
      return {
        unread: Math.max(0, current.unread - flipped),
        items: current.items.map((event) =>
          targets.has(event.id) && !event.read_at
            ? { ...event, read_at: stamp }
            : event,
        ),
      };
    });
    startTransition(async () => {
      try {
        await api.markAlertsRead(ids, pin);
      } catch (cause) {
        setEvents(snapshot);
        report(cause);
      }
    });
  }

  function markAllRead() {
    if (!pin) return;
    setError(null);
    const snapshot = events;
    const stamp = new Date().toISOString();
    setEvents((current) =>
      current
        ? {
            unread: 0,
            items: current.items.map((event) =>
              event.read_at ? event : { ...event, read_at: stamp },
            ),
          }
        : current,
    );
    startTransition(async () => {
      try {
        await api.markAllAlertsRead(pin);
      } catch (cause) {
        setEvents(snapshot);
        report(cause);
      }
    });
  }

  async function saveRule(request: AlertRuleMutation, editingId: number | null) {
    setError(null);
    setSaving(true);
    try {
      if (editingId) await api.updateAlertRule(editingId, request, pin);
      else await api.createAlertRule(request, pin);
      setLoading(true);
      await load();
      await refreshEvents();
    } catch (cause) {
      report(cause);
      throw cause;
    } finally {
      setSaving(false);
    }
  }

  function deleteRule(rule: AlertRule) {
    if (!window.confirm(`Delete ${rule.name}?`)) return;
    setError(null);
    startTransition(async () => {
      try {
        await api.deleteAlertRule(rule.id, pin);
        setLoading(true);
        await load();
      } catch (cause) {
        report(cause);
      }
    });
  }

  function toggleRule(rule: AlertRule) {
    setError(null);
    startTransition(async () => {
      try {
        await api.updateAlertRule(
          rule.id,
          {
            name: rule.name,
            enabled: !rule.enabled,
            watchlist_id: rule.watchlist_id,
            query: rule.query,
            notify_discord: rule.notify_discord,
            notify_email: rule.notify_email,
          },
          pin,
        );
        await load();
      } catch (cause) {
        report(cause);
      }
    });
  }

  const unread = events?.unread ?? 0;
  const activeRules = (rules ?? []).filter((rule) => rule.enabled).length;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Alerts</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            Rules run against persisted snapshots
            {status ? ` every ${status.evaluation_interval_minutes} minutes` : ""} and
            drop their matches here. Events are threaded by ticker so one busy
            snapshot reads as a single line, not fifty. Shared workspace: changing
            rules or clearing events needs the PIN.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Chip
            label={
              unread ? `${unread} unread` : eventsLoading ? "Loading..." : "Inbox clear"
            }
            tone={unread ? "good" : "muted"}
          />
          <Chip
            label={`Scheduler ${status ? (status.scheduler_enabled ? "on" : "off") : "..."}`}
            tone={status ? (status.scheduler_enabled ? "good" : "bad") : "muted"}
          />
          <Chip
            label={`Discord ${status?.discord_configured ? "on" : "off"}`}
            tone={status?.discord_configured ? "good" : "muted"}
          />
          <Chip
            label={`Email ${status?.email_configured ? "on" : "off"}`}
            tone={status?.email_configured ? "good" : "muted"}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border">
        <Tabs
          tabs={[
            { id: "inbox", label: unread ? `Inbox (${unread})` : "Inbox" },
            { id: "rules", label: `Rules (${activeRules})` },
          ]}
          active={tab}
          onChange={setTab}
        />
        <label className="flex items-center gap-2 pb-2 text-xs text-muted">
          <span className={pin ? "text-positive" : "text-faint"}>
            {pin ? "Unlocked" : "Locked"}
          </span>
          <input
            type="password"
            value={pin}
            onChange={(event) => setPin(event.target.value)}
            maxLength={128}
            autoComplete="current-password"
            placeholder="Workspace PIN"
            className="w-40 rounded border border-border bg-surface px-2 py-1 text-sm text-foreground placeholder:text-faint"
          />
        </label>
      </div>

      {error && (
        <p className="rounded border border-negative/50 px-3 py-2 text-sm text-negative">
          {error}
        </p>
      )}

      {tab === "inbox" ? (
        <AlertInbox
          events={events?.items ?? []}
          unread={unread}
          loading={eventsLoading}
          rules={rules ?? []}
          canTriage={Boolean(pin)}
          busy={pending}
          onMarkRead={markRead}
          onMarkAllRead={markAllRead}
        />
      ) : (
        <AlertRulesPanel
          rules={rules}
          watchlists={watchlists?.items ?? []}
          fields={fields}
          status={status}
          loading={loading}
          pending={pending || saving}
          canMutate={Boolean(pin)}
          onSave={saveRule}
          onDelete={deleteRule}
          onToggle={toggleRule}
        />
      )}
    </div>
  );
}
