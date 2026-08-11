"use client";

import { useState } from "react";
import FilterBuilder from "@/components/screener/FilterBuilder";
import Panel from "@/components/ui/Panel";
import {
  AlertRule,
  AlertRuleMutation,
  AlertStatus,
  CustomScreenerRequest,
  ScreenerFieldSpec,
  Watchlist,
} from "@/lib/api";

// Rule shelf: the saved rules on the left, the editor opening only when you
// create or edit one, so the page is not permanently occupied by a form.

const DEFAULT_QUERY: CustomScreenerRequest = {
  scope: "contract",
  conditions: [{ field: "volume_oi", operator: "gte", value: 3 }],
  sort: { field: "volume_oi", direction: "desc" },
  limit: 25,
};

export default function AlertRulesPanel({
  rules,
  watchlists,
  fields,
  status,
  loading,
  pending,
  canMutate,
  onSave,
  onDelete,
  onToggle,
}: {
  rules: AlertRule[] | null;
  watchlists: Watchlist[];
  fields: ScreenerFieldSpec[];
  status: AlertStatus | null;
  loading: boolean;
  pending: boolean;
  canMutate: boolean;
  onSave: (request: AlertRuleMutation, editingId: number | null) => Promise<void>;
  onDelete: (rule: AlertRule) => void;
  onToggle: (rule: AlertRule) => void;
}) {
  const [open, setOpen] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [watchlistId, setWatchlistId] = useState(0);
  const [query, setQuery] = useState<CustomScreenerRequest>(DEFAULT_QUERY);
  const [enabled, setEnabled] = useState(true);
  const [discord, setDiscord] = useState(false);
  const [email, setEmail] = useState(false);

  function openNew() {
    setEditingId(null);
    setName("");
    setQuery(DEFAULT_QUERY);
    setEnabled(true);
    setDiscord(false);
    setEmail(false);
    setWatchlistId(watchlists[0]?.id ?? 0);
    setOpen(true);
  }

  function openEdit(rule: AlertRule) {
    setEditingId(rule.id);
    setName(rule.name);
    setWatchlistId(rule.watchlist_id);
    setQuery(rule.query);
    setEnabled(rule.enabled);
    setDiscord(rule.notify_discord);
    setEmail(rule.notify_email);
    setOpen(true);
  }

  async function submit() {
    try {
      await onSave(
        {
          name,
          enabled,
          watchlist_id: watchlistId,
          query,
          notify_discord: discord,
          notify_email: email,
        },
        editingId,
      );
    } catch {
      return; // The page surfaces the error; keep the editor open to retry.
    }
    setOpen(false);
    setEditingId(null);
  }

  const editingRule = rules?.find((rule) => rule.id === editingId) ?? null;

  return (
    <div className="space-y-3">
      <Panel
        title="Rules"
        right={
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-muted">
              {loading ? "..." : `${rules?.length ?? 0}`}
            </span>
            <button
              type="button"
              onClick={openNew}
              disabled={loading || !watchlists.length}
              className="rounded border border-accent bg-accent/15 px-2.5 py-1 text-xs text-accent-strong disabled:opacity-40"
            >
              New rule
            </button>
          </div>
        }
        bodyClassName="p-0"
      >
        <div className="divide-y divide-border">
          {loading && (
            <p className="px-4 py-10 text-center text-sm text-muted">
              Loading alert rules...
            </p>
          )}
          {!loading && !watchlists.length && (
            <p className="px-4 py-6 text-center text-sm text-amber-200">
              Create a watchlist before creating an alert rule.
            </p>
          )}
          {(rules ?? []).map((rule) => (
            <div
              key={rule.id}
              className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{rule.name}</span>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] uppercase ${
                      rule.enabled ? "bg-positive/10 text-positive" : "bg-border text-faint"
                    }`}
                  >
                    {rule.enabled ? "Active" : "Paused"}
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted">
                  {rule.watchlist_name} · {rule.query.scope} ·{" "}
                  {rule.query.conditions.length} condition
                  {rule.query.conditions.length === 1 ? "" : "s"}
                  {rule.notify_discord ? " · Discord" : ""}
                  {rule.notify_email ? " · Email" : ""}
                </p>
                {rule.last_error && (
                  <p className="mt-1 text-xs text-negative">{rule.last_error}</p>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={pending || !canMutate}
                  onClick={() => onToggle(rule)}
                  className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40"
                >
                  {rule.enabled ? "Pause" : "Resume"}
                </button>
                <button
                  type="button"
                  disabled={pending}
                  onClick={() => openEdit(rule)}
                  className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40"
                >
                  Edit
                </button>
                <button
                  type="button"
                  disabled={pending || !canMutate}
                  onClick={() => onDelete(rule)}
                  className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative disabled:opacity-40"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
          {!loading && rules && !rules.length && watchlists.length > 0 && (
            <p className="px-4 py-10 text-center text-sm text-muted">
              No alert rules yet. Create one to start filling the inbox.
            </p>
          )}
        </div>
      </Panel>

      {open && (
        <Panel
          title={editingRule ? `Edit ${editingRule.name}` : "New alert rule"}
          right={
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded border border-border px-2 py-1 text-xs text-muted"
            >
              Close
            </button>
          }
        >
          <fieldset disabled={pending || loading}>
            <div className="grid gap-3 md:grid-cols-2">
              <label className="text-xs text-muted">
                Name
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  maxLength={64}
                  className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-foreground"
                />
              </label>
              <label className="text-xs text-muted">
                Watchlist
                <select
                  value={watchlistId}
                  onChange={(event) => setWatchlistId(Number(event.target.value))}
                  className="mt-1 w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-foreground"
                >
                  {watchlists.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="mt-4">
              {loading ? (
                <p className="text-sm text-muted">Loading alert fields...</p>
              ) : (
                <FilterBuilder fields={fields} value={query} onChange={setQuery} />
              )}
            </div>

            <div className="mt-4 flex flex-wrap gap-4 text-sm">
              <label>
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(event) => setEnabled(event.target.checked)}
                  className="mr-2"
                />
                Enabled
              </label>
              <label className={!status?.discord_configured ? "text-faint" : ""}>
                <input
                  type="checkbox"
                  checked={discord}
                  disabled={!status?.discord_configured}
                  onChange={(event) => setDiscord(event.target.checked)}
                  className="mr-2"
                />
                Discord
              </label>
              <label className={!status?.email_configured ? "text-faint" : ""}>
                <input
                  type="checkbox"
                  checked={email}
                  disabled={!status?.email_configured}
                  onChange={(event) => setEmail(event.target.checked)}
                  className="mr-2"
                />
                Email
              </label>
            </div>

            <div className="mt-4 flex gap-2">
              <button
                type="button"
                onClick={submit}
                disabled={
                  !canMutate || !name.trim() || !watchlistId || !query.conditions.length
                }
                className="rounded border border-accent bg-accent/15 px-3 py-1.5 text-sm text-accent-strong disabled:opacity-40"
              >
                {pending ? "Saving..." : editingRule ? "Update rule" : "Create rule"}
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded border border-border px-3 py-1.5 text-sm text-muted"
              >
                Cancel
              </button>
            </div>
          </fieldset>
        </Panel>
      )}
    </div>
  );
}
