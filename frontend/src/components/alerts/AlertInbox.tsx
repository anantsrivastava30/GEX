"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { AlertEvent, AlertRule } from "@/lib/api";

// Triage inbox for alert events. Raw events are noisy - one rule firing on a
// busy snapshot emits a row per matched contract - so events are threaded by
// ticker and snapshot date, the way a mail client threads a conversation. The
// list stays scannable; the matched rows live in the detail pane.

export interface AlertThread {
  key: string;
  symbol: string;
  snapshotDate: string;
  events: AlertEvent[];
  unread: number;
  latest: string;
  rules: string[];
}

function threadOf(events: AlertEvent[]): AlertThread[] {
  const groups = new Map<string, AlertEvent[]>();
  for (const event of events) {
    const key = `${event.snapshot_date}|${event.symbol}`;
    const bucket = groups.get(key);
    if (bucket) bucket.push(event);
    else groups.set(key, [event]);
  }
  const threads = [...groups.entries()].map(([key, items]) => {
    const sorted = [...items].sort((a, b) => b.created_at.localeCompare(a.created_at));
    return {
      key,
      symbol: sorted[0].symbol,
      snapshotDate: sorted[0].snapshot_date,
      events: sorted,
      unread: sorted.filter((event) => !event.read_at).length,
      latest: sorted[0].created_at,
      rules: [...new Set(sorted.map((event) => event.rule_name))],
    };
  });
  return threads.sort((a, b) => b.latest.localeCompare(a.latest));
}

function dayLabel(snapshotDate: string) {
  const today = new Date();
  const iso = (date: Date) => date.toISOString().slice(0, 10);
  if (snapshotDate === iso(today)) return "Today";
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (snapshotDate === iso(yesterday)) return "Yesterday";
  const parsed = new Date(`${snapshotDate}T00:00:00`);
  return Number.isNaN(parsed.getTime())
    ? snapshotDate
    : parsed.toLocaleDateString(undefined, {
        weekday: "short",
        month: "short",
        day: "numeric",
      });
}

function relativeTime(iso: string) {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso.slice(0, 10);
  const minutes = Math.round((Date.now() - then) / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d`;
  return new Date(then).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function formatValue(value: unknown) {
  if (value == null || value === "") return "—";
  if (typeof value === "number") {
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

function deliveryTone(state: string) {
  if (state === "sent") return "text-positive";
  if (state === "failed" || state === "error") return "text-negative";
  if (state === "pending") return "text-warning";
  return "text-faint";
}

function matchedValues(event: AlertEvent): [string, unknown][] {
  const payload = event.payload as { values?: Record<string, unknown> };
  const values = payload?.values;
  if (!values || typeof values !== "object") return [];
  return Object.entries(values).filter(
    ([key]) => key !== "symbol" && key !== "snapshot_date",
  );
}

function EventCard({
  event,
  canTriage,
  busy,
  onMarkRead,
}: {
  event: AlertEvent;
  canTriage: boolean;
  busy: boolean;
  onMarkRead: (ids: number[]) => void;
}) {
  const values = matchedValues(event);
  return (
    <article className="rounded-md border border-border bg-surface-2 px-3 py-2.5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm text-foreground">{event.message}</p>
          <p className="mt-1 text-[11px] text-faint">
            {event.rule_name} · {event.scope} · {relativeTime(event.created_at)} ago
          </p>
        </div>
        {event.read_at ? (
          <span className="whitespace-nowrap text-[11px] text-faint">read</span>
        ) : (
          <button
            type="button"
            disabled={!canTriage || busy}
            onClick={() => onMarkRead([event.id])}
            className="whitespace-nowrap rounded border border-border px-2 py-0.5 text-[11px] text-muted hover:text-foreground disabled:opacity-40"
          >
            Mark read
          </button>
        )}
      </div>

      {values.length > 0 && (
        <dl className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-border pt-2 sm:grid-cols-3">
          {values.map(([key, value]) => (
            <div key={key} className="min-w-0">
              <dt className="truncate text-[11px] uppercase tracking-wide text-faint">
                {key.replace(/_/g, " ")}
              </dt>
              <dd className="truncate font-mono text-xs text-foreground">
                {formatValue(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {(event.discord_status !== "not_requested" ||
        event.email_status !== "not_requested") && (
        <p className="mt-2 flex gap-3 text-[11px]">
          {event.discord_status !== "not_requested" && (
            <span className={deliveryTone(event.discord_status)}>
              Discord {event.discord_status}
            </span>
          )}
          {event.email_status !== "not_requested" && (
            <span className={deliveryTone(event.email_status)}>
              Email {event.email_status}
            </span>
          )}
        </p>
      )}
    </article>
  );
}

export default function AlertInbox({
  events,
  unread,
  loading,
  rules,
  canTriage,
  busy,
  onMarkRead,
  onMarkAllRead,
}: {
  events: AlertEvent[];
  unread: number;
  loading: boolean;
  rules: AlertRule[];
  canTriage: boolean;
  busy: boolean;
  onMarkRead: (ids: number[]) => void;
  onMarkAllRead: () => void;
}) {
  const [search, setSearch] = useState("");
  const [ruleFilter, setRuleFilter] = useState(0);
  const [unreadOnly, setUnreadOnly] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const threads = useMemo(() => {
    const needle = search.trim().toUpperCase();
    const filtered = events.filter((event) => {
      if (unreadOnly && event.read_at) return false;
      if (ruleFilter && event.alert_rule_id !== ruleFilter) return false;
      if (!needle) return true;
      return (
        event.symbol.toUpperCase().includes(needle) ||
        event.rule_name.toUpperCase().includes(needle) ||
        event.message.toUpperCase().includes(needle)
      );
    });
    return threadOf(filtered);
  }, [events, search, ruleFilter, unreadOnly]);

  // Derived, so a selection filtered out of view simply falls back to the
  // list and comes back if the thread returns.
  const selected = threads.find((thread) => thread.key === selectedKey) ?? null;

  useEffect(() => {
    function onKey(nativeEvent: KeyboardEvent) {
      const target = nativeEvent.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);
      if (nativeEvent.key === "/" && !typing) {
        nativeEvent.preventDefault();
        searchRef.current?.focus();
        return;
      }
      if (nativeEvent.key === "Escape") {
        if (typing) (target as HTMLElement).blur();
        else setSelectedKey(null);
        return;
      }
      if (typing || nativeEvent.metaKey || nativeEvent.ctrlKey) return;
      if (nativeEvent.key === "j" || nativeEvent.key === "k") {
        if (!threads.length) return;
        nativeEvent.preventDefault();
        const index = threads.findIndex((thread) => thread.key === selectedKey);
        const step = nativeEvent.key === "j" ? 1 : -1;
        const next =
          index === -1
            ? nativeEvent.key === "j"
              ? 0
              : threads.length - 1
            : Math.min(Math.max(index + step, 0), threads.length - 1);
        setSelectedKey(threads[next].key);
        return;
      }
      if (nativeEvent.key === "u") {
        setUnreadOnly((current) => !current);
        return;
      }
      if (nativeEvent.key === "e" && selectedKey) {
        const thread = threads.find((item) => item.key === selectedKey);
        if (!thread || !canTriage || busy) return;
        const ids = thread.events.filter((item) => !item.read_at).map((item) => item.id);
        if (ids.length) onMarkRead(ids);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [threads, selectedKey, canTriage, busy, onMarkRead]);

  const sections: { date: string; threads: AlertThread[] }[] = [];
  for (const thread of threads) {
    const last = sections[sections.length - 1];
    if (last && last.date === thread.snapshotDate) last.threads.push(thread);
    else sections.push({ date: thread.snapshotDate, threads: [thread] });
  }

  const visibleUnread = threads.reduce((total, thread) => total + thread.unread, 0);

  return (
    <div className="rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2.5">
        <input
          ref={searchRef}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Filter by ticker, rule, or text  (/)"
          className="min-w-[12rem] flex-1 rounded border border-border bg-surface-2 px-2.5 py-1.5 text-sm text-foreground placeholder:text-faint"
        />
        <select
          value={ruleFilter}
          onChange={(event) => setRuleFilter(Number(event.target.value))}
          className="rounded border border-border bg-surface-2 px-2 py-1.5 text-xs text-muted"
        >
          <option value={0}>All rules</option>
          {rules.map((rule) => (
            <option key={rule.id} value={rule.id}>
              {rule.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setUnreadOnly((current) => !current)}
          className={`rounded border px-2.5 py-1.5 text-xs ${
            unreadOnly
              ? "border-accent bg-accent/15 text-accent-strong"
              : "border-border text-muted hover:text-foreground"
          }`}
        >
          Unread{unread ? ` ${unread}` : ""}
        </button>
        <button
          type="button"
          disabled={!canTriage || busy || !unread}
          onClick={onMarkAllRead}
          className="rounded border border-border px-2.5 py-1.5 text-xs text-muted hover:text-foreground disabled:opacity-40"
        >
          Mark all read
        </button>
      </div>

      <div className="grid lg:grid-cols-[21rem_minmax(0,1fr)]">
        <div
          className={`max-h-[36rem] overflow-auto border-border lg:border-r ${
            selected ? "hidden lg:block" : "block"
          }`}
        >
          {loading && (
            <p role="status" className="px-4 py-12 text-center text-sm text-muted">
              Loading alert events...
            </p>
          )}
          {!loading && !threads.length && (
            <p className="px-4 py-12 text-center text-sm text-muted">
              {events.length
                ? "No events match these filters."
                : "Nothing in the inbox. Events appear after a snapshot matches an enabled rule."}
            </p>
          )}
          {sections.map((section) => (
            <div key={section.date}>
              <p className="sticky top-0 z-10 border-b border-border bg-surface/95 px-3 py-1.5 text-[11px] uppercase tracking-wide text-faint backdrop-blur">
                {dayLabel(section.date)}
              </p>
              {section.threads.map((thread) => {
                const isActive = thread.key === selectedKey;
                return (
                  <button
                    key={thread.key}
                    type="button"
                    onClick={() => setSelectedKey(isActive ? null : thread.key)}
                    className={`flex w-full gap-2.5 border-b border-border px-3 py-2.5 text-left transition-colors ${
                      isActive ? "bg-surface-hover" : "hover:bg-surface-hover"
                    }`}
                  >
                    <span
                      aria-hidden
                      className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${
                        thread.unread ? "bg-accent" : "bg-transparent"
                      }`}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-baseline justify-between gap-2">
                        <span
                          className={`font-mono text-sm ${
                            thread.unread ? "text-foreground" : "text-muted"
                          }`}
                        >
                          {thread.symbol}
                          {thread.events.length > 1 && (
                            <span className="ml-1.5 rounded bg-border px-1 py-0.5 font-sans text-[10px] text-muted">
                              {thread.events.length}
                            </span>
                          )}
                        </span>
                        <span className="whitespace-nowrap font-mono text-[11px] text-faint">
                          {relativeTime(thread.latest)}
                        </span>
                      </span>
                      <span className="mt-0.5 block truncate text-xs text-muted">
                        {thread.rules.join(", ")}
                      </span>
                      <span className="mt-0.5 block truncate text-[11px] text-faint">
                        {thread.events[0].message}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        <div className={`${selected ? "block" : "hidden lg:block"}`}>
          {!selected && (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <p className="text-sm text-muted">
                Select a ticker to inspect the rows that matched.
              </p>
              <p className="text-xs text-faint">
                <span className="font-mono">j</span>/<span className="font-mono">k</span>{" "}
                move · <span className="font-mono">e</span> mark thread read ·{" "}
                <span className="font-mono">u</span> unread only ·{" "}
                <span className="font-mono">/</span> search
              </p>
            </div>
          )}
          {selected && (
            <div className="max-h-[36rem] overflow-auto">
              <div className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-2 border-b border-border bg-surface/95 px-4 py-2.5 backdrop-blur">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSelectedKey(null)}
                    className="rounded border border-border px-2 py-1 text-xs text-muted lg:hidden"
                  >
                    Back
                  </button>
                  <Link
                    href={`/stock/${selected.symbol}`}
                    className="font-mono text-sm text-accent hover:underline"
                  >
                    {selected.symbol}
                  </Link>
                  <span className="text-xs text-muted">
                    {selected.events.length} match
                    {selected.events.length === 1 ? "" : "es"} ·{" "}
                    {dayLabel(selected.snapshotDate)}
                  </span>
                </div>
                <button
                  type="button"
                  disabled={!canTriage || busy || !selected.unread}
                  onClick={() =>
                    onMarkRead(
                      selected.events.filter((item) => !item.read_at).map((item) => item.id),
                    )
                  }
                  className="rounded border border-border px-2.5 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40"
                >
                  {selected.unread ? `Mark ${selected.unread} read` : "All read"}
                </button>
              </div>
              <div className="space-y-2 p-3">
                {selected.events.map((event) => (
                  <EventCard
                    key={event.id}
                    event={event}
                    canTriage={canTriage}
                    busy={busy}
                    onMarkRead={onMarkRead}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-3 py-2 text-[11px] text-faint">
        <span>
          {threads.length} thread{threads.length === 1 ? "" : "s"} · {visibleUnread} unread
          in view
        </span>
        {!canTriage && <span>Enter the workspace PIN to triage events.</span>}
      </div>
    </div>
  );
}
