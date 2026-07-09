export default function FlowPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Flow — Unusual Activity</h1>
      <p className="max-w-2xl text-sm text-muted">
        The proxy flow feed (volume/open-interest anomalies, OI change from our
        daily snapshots, hottest chains) lands here in Phase 2. The backend
        endpoint already exists at{" "}
        <code className="font-mono">/api/flow/unusual</code>; a true trade-tape
        feed slots in once a paid OPRA data provider is connected (Phase 4).
      </p>
    </div>
  );
}
