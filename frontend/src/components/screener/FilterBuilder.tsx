"use client";

import {
  CustomScreenerRequest,
  ScreenerCondition,
  ScreenerFieldSpec,
  ScreenerScope,
} from "@/lib/api";

const OPERATOR_LABELS = {
  eq: "=",
  ne: "!=",
  gt: ">",
  gte: ">=",
  lt: "<",
  lte: "<=",
};

function defaultValue(field?: ScreenerFieldSpec): string | number | boolean {
  if (field?.type === "boolean") return true;
  if (field?.example != null) return field.example;
  if (field?.choices?.length) return field.choices[0];
  if (field?.type === "number") return "";
  return "";
}

function preferredField(fields: ScreenerFieldSpec[], scope: ScreenerScope) {
  const preferred = scope === "contract" ? "volume_oi" : "gamma_gap_score";
  return fields.find((field) => field.name === preferred && field.scopes.includes(scope))
    ?? fields.find((field) => field.scopes.includes(scope));
}

export default function FilterBuilder({
  fields,
  value,
  onChange,
  showScope = true,
}: {
  fields: ScreenerFieldSpec[];
  value: CustomScreenerRequest;
  onChange: (value: CustomScreenerRequest) => void;
  showScope?: boolean;
}) {
  const available = fields.filter((field) => field.scopes.includes(value.scope));
  const sortable = available;

  function changeScope(scope: ScreenerScope) {
    const first = preferredField(fields, scope);
    onChange({
      scope,
      conditions: first
        ? [{ field: first.name, operator: first.operators[0], value: defaultValue(first) }]
        : [],
      sort: first ? { field: first.name, direction: "desc" } : undefined,
      limit: value.limit,
    });
  }

  function updateCondition(index: number, condition: ScreenerCondition) {
    onChange({
      ...value,
      conditions: value.conditions.map((item, itemIndex) =>
        itemIndex === index ? condition : item,
      ),
    });
  }

  function addCondition() {
    const field = available.find(
      (candidate) => !value.conditions.some((condition) => condition.field === candidate.name),
    ) ?? available[0];
    if (!field || value.conditions.length >= 12) return;
    onChange({
      ...value,
      conditions: [
        ...value.conditions,
        {
          field: field.name,
          operator: field.operators[0],
          value: defaultValue(field),
        },
      ],
    });
  }

  return (
    <div className="space-y-3">
      {showScope && (
        <div className="flex gap-2">
          {(["contract", "ticker"] as ScreenerScope[]).map((scope) => (
            <button
              key={scope}
              type="button"
              aria-pressed={value.scope === scope}
              onClick={() => changeScope(scope)}
              className={`rounded-full border px-3 py-1 text-xs capitalize ${
                value.scope === scope
                  ? "border-accent bg-accent/15 text-foreground"
                  : "border-border text-muted hover:text-foreground"
              }`}
            >
              {scope}s
            </button>
          ))}
        </div>
      )}

      <p className="text-xs text-muted">
        Every condition below must match. Sorting only controls which matches appear first.
      </p>

      <div className="space-y-2">
        {value.conditions.map((condition, index) => {
          const field = available.find((item) => item.name === condition.field);
          return (
            <div key={`${index}-${condition.field}`} className="space-y-1">
              <div className="grid gap-2 sm:grid-cols-[minmax(10rem,1fr)_5rem_minmax(8rem,1fr)_auto]">
              <select
                aria-label={`Condition ${index + 1} field`}
                value={condition.field}
                onChange={(event) => {
                  const nextField = available.find(
                    (item) => item.name === event.target.value,
                  );
                  if (!nextField) return;
                  updateCondition(index, {
                    field: nextField.name,
                    operator: nextField.operators[0],
                    value: defaultValue(nextField),
                  });
                }}
                className="rounded border border-border bg-surface px-2 py-1.5 text-sm"
              >
                {available.map((item) => (
                  <option key={item.name} value={item.name}>
                    {item.label}
                  </option>
                ))}
              </select>
              <select
                aria-label={`Condition ${index + 1} operator`}
                value={condition.operator}
                onChange={(event) =>
                  updateCondition(index, {
                    ...condition,
                    operator: event.target.value as ScreenerCondition["operator"],
                  })
                }
                className="rounded border border-border bg-surface px-2 py-1.5 text-sm"
              >
                {(field?.operators ?? []).map((operator) => (
                  <option key={operator} value={operator}>
                    {OPERATOR_LABELS[operator]}
                  </option>
                ))}
              </select>
              {field?.type === "boolean" ? (
                <select
                  aria-label={`Condition ${index + 1} value`}
                  value={String(condition.value)}
                  onChange={(event) =>
                    updateCondition(index, {
                      ...condition,
                      value: event.target.value === "true",
                    })
                  }
                  className="rounded border border-border bg-surface px-2 py-1.5 text-sm"
                >
                  <option value="true">True</option>
                  <option value="false">False</option>
                </select>
              ) : field?.choices?.length ? (
                <select
                  aria-label={`Condition ${index + 1} value`}
                  value={String(condition.value)}
                  onChange={(event) => updateCondition(index, { ...condition, value: event.target.value })}
                  className="rounded border border-border bg-surface px-2 py-1.5 text-sm"
                >
                  {field.choices.map((choice) => <option key={choice} value={choice}>{choice.toUpperCase()}</option>)}
                </select>
              ) : (
                <input
                  aria-label={`Condition ${index + 1} value`}
                  type={field?.type === "number" ? "number" : field?.type === "date" ? "date" : "text"}
                  step={field?.type === "number" ? "any" : undefined}
                  placeholder={field?.example == null ? undefined : String(field.example)}
                  value={String(condition.value)}
                  onChange={(event) =>
                    updateCondition(index, {
                      ...condition,
                        value:
                          field?.type === "number"
                          ? event.target.value === "" ? "" : Number(event.target.value)
                          : event.target.value,
                    })
                  }
                  className="rounded border border-border bg-surface px-2 py-1.5 text-sm"
                />
              )}
              <button
                type="button"
                onClick={() =>
                  onChange({
                    ...value,
                    conditions: value.conditions.filter((_, itemIndex) => itemIndex !== index),
                  })
                }
                className="rounded border border-border px-2 py-1 text-xs text-muted hover:text-negative"
              >
                Remove
              </button>
              </div>
              {(field?.description || field?.requires_history) && (
                <p className="text-[12px] text-faint">
                  {field.description}{field.history_requirement ? ` ${field.history_requirement}` : field.requires_history ? " Requires two consecutive market-session snapshots." : ""}
                </p>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={addCondition}
          disabled={!available.length || value.conditions.length >= 12}
          className="rounded border border-border px-3 py-1 text-xs text-muted hover:text-foreground disabled:opacity-40"
        >
          Add condition
        </button>
        <span className="ml-1 text-xs text-faint">Sort</span>
        <select
          aria-label="Sort field"
          value={value.sort?.field ?? ""}
          onChange={(event) =>
            onChange({
              ...value,
              sort: event.target.value
                ? { field: event.target.value, direction: value.sort?.direction ?? "desc" }
                : undefined,
            })
          }
          className="rounded border border-border bg-surface px-2 py-1 text-xs"
        >
          <option value="">Default</option>
          {sortable.map((field) => (
            <option key={field.name} value={field.name}>
              {field.label}
            </option>
          ))}
        </select>
        <select
          aria-label="Sort direction"
          value={value.sort?.direction ?? "desc"}
          disabled={!value.sort}
          onChange={(event) =>
            value.sort &&
            onChange({
              ...value,
              sort: { ...value.sort, direction: event.target.value as "asc" | "desc" },
            })
          }
          className="rounded border border-border bg-surface px-2 py-1 text-xs disabled:opacity-40"
        >
          <option value="desc">Descending</option>
          <option value="asc">Ascending</option>
        </select>
      </div>
    </div>
  );
}
