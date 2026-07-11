import Panel from "@/components/ui/Panel";

const CALENDAR_URL =
  "https://sslecal2.investing.com?columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&importance=2,3&features=datepicker,timezone,filters&countries=5&calType=week&timeZone=8&lang=1";
const CALENDAR_PAGE_URL = "https://www.investing.com/economic-calendar/";

export default function CalendarPage() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">Economic Calendar</h1>
        <p className="text-sm text-muted">
          Live third-party economic calendar content provided by Investing.com.
        </p>
      </div>

      <Panel
        title="High-impact releases"
        right={
          <a
            href={CALENDAR_PAGE_URL}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-accent hover:text-accent-strong"
          >
            Open on Investing.com
          </a>
        }
        bodyClassName="p-0"
      >
        <iframe
          title="Investing.com economic calendar"
          src={CALENDAR_URL}
          className="min-h-[850px] w-full border-0"
          loading="lazy"
        >
          <p>
            The embedded economic calendar is unavailable. Visit{" "}
            <a href={CALENDAR_PAGE_URL}>Investing.com&apos;s economic calendar</a>.
          </p>
        </iframe>
        <p className="border-t border-border px-4 py-2 text-xs text-muted">
          If the calendar does not load, use the{" "}
          <a
            href={CALENDAR_PAGE_URL}
            target="_blank"
            rel="noreferrer"
            className="text-accent hover:text-accent-strong"
          >
            direct Investing.com calendar link
          </a>
          .
        </p>
      </Panel>
    </div>
  );
}
