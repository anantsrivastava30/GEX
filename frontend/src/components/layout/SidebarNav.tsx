"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState, type ComponentType } from "react";
import {
  IconAI,
  IconAlerts,
  IconCalendar,
  IconCongress,
  IconFlow,
  IconMarket,
  IconMore,
  IconNews,
  IconScreener,
  IconTicker,
  IconTrackRecord,
  IconTools,
  IconWatchlist,
} from "./icons";

type NavItem = {
  href: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  match?: string;
  soon?: boolean;
};

const SECTIONS: { heading: string; items: NavItem[] }[] = [
  {
    heading: "Markets",
    items: [
      { href: "/market", label: "Market", icon: IconMarket },
      { href: "/flow", label: "Flow", icon: IconFlow },
      { href: "/stock/SPY", label: "Tickers", icon: IconTicker, match: "/stock" },
      { href: "/screener", label: "Screener", icon: IconScreener },
      { href: "/watchlists", label: "Watchlists", icon: IconWatchlist },
      { href: "/alerts", label: "Alerts", icon: IconAlerts },
    ],
  },
  {
    heading: "Research",
    items: [
      { href: "/news", label: "News", icon: IconNews },
      { href: "/track-record", label: "Track Record", icon: IconTrackRecord },
      { href: "/ai", label: "AI Analysis", icon: IconAI },
      { href: "/calendar", label: "Calendar", icon: IconCalendar },
      { href: "/congress", label: "Congress", icon: IconCongress },
      { href: "/tools/binomial", label: "Tools", icon: IconTools },
    ],
  },
];

// Phone tab bar: a handful of primary destinations, the rest behind "More".
// Twelve items in one scroller hid over half of them off-screen.
const MOBILE_PRIMARY = ["/market", "/flow", "/stock/SPY", "/screener"];

export default function SidebarNav() {
  const pathname = usePathname();
  const [moreOpen, setMoreOpen] = useState(false);
  const sheetRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const allItems = SECTIONS.flatMap((section) => section.items);
  const primaryItems = MOBILE_PRIMARY.map(
    (href) => allItems.find((item) => item.href === href)!,
  ).filter(Boolean);
  const overflowItems = allItems.filter(
    (item) => !MOBILE_PRIMARY.includes(item.href),
  );
  const overflowActive = overflowItems.some((item) =>
    pathname.startsWith(item.match ?? item.href),
  );

  // The sheet is a modal dialog, but aria-modal alone does not move, trap, or
  // restore focus, so do it here: focus the first entry on open, keep Tab
  // inside the sheet while it is open, and hand focus back to the trigger on
  // close. Escape closes it; every link closes it from its own handler.
  useEffect(() => {
    const sheet = sheetRef.current;
    if (!moreOpen || !sheet) return;
    const trigger = triggerRef.current;

    const focusable = () =>
      Array.from(sheet.querySelectorAll<HTMLElement>("a[href], button:not([disabled])"));
    focusable()[0]?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMoreOpen(false);
        return;
      }
      if (e.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (!sheet.contains(active)) {
        e.preventDefault();
        first.focus();
      } else if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      trigger?.focus();
    };
  }, [moreOpen]);

  return (
    <>
    <aside className="fixed inset-y-0 left-0 z-20 hidden w-52 flex-col border-r border-border bg-surface md:flex">
      <Link
        href="/market"
        className="flex items-center gap-2 px-4 py-4 text-foreground"
      >
        <span className="grid h-7 w-7 place-items-center rounded-md bg-accent/15 text-accent">
          <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4" aria-hidden>
            <path d="M3 13c2.5 0 2.5 2 5 2s2.5-2 5-2 2.5 2 5 2 2.5-2 3-2v3c-.5 0-.5 2-3 2s-2.5-2-5-2-2.5 2-5 2-2.5-2-5-2v-3zm14-3.5c0 1.4-1.1 2.5-2.5 2.5S12 10.9 12 9.5 14.5 4 14.5 4 17 8.1 17 9.5z" />
          </svg>
        </span>
        <span className="font-semibold tracking-tight">GEX Terminal</span>
      </Link>

      <nav className="flex-1 overflow-y-auto px-2 py-1">
        {SECTIONS.map((section) => (
          <div key={section.heading} className="mb-3">
            <p className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-faint">
              {section.heading}
            </p>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const active = pathname.startsWith(item.match ?? item.href);
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`group flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                      active
                        ? "bg-accent/15 text-foreground"
                        : "text-muted hover:bg-surface-hover hover:text-foreground"
                    }`}
                  >
                    <Icon
                      className={`h-4 w-4 shrink-0 ${
                        active ? "text-accent" : "text-faint group-hover:text-muted"
                      }`}
                    />
                    {item.label}
                    {item.soon && (
                      <span className="ml-auto rounded bg-border px-1.5 py-0.5 text-[9px] uppercase text-faint">
                        soon
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-border px-4 py-3 text-[11px] leading-4 text-faint">
        Research &amp; education only. Not financial advice.
      </div>
    </aside>
    {moreOpen && (
      <>
        <button
          type="button"
          tabIndex={-1}
          aria-hidden
          onClick={() => setMoreOpen(false)}
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
        />
        <div
          ref={sheetRef}
          role="dialog"
          aria-modal="true"
          aria-label="More destinations"
          className="fixed inset-x-0 bottom-16 z-40 max-h-[60vh] overflow-y-auto rounded-t-xl border-t border-border bg-surface pb-2 md:hidden"
        >
          <p className="px-4 pb-1 pt-3 text-xs font-semibold uppercase tracking-wider text-faint">
            More
          </p>
          {overflowItems.map((item) => {
            const active = pathname.startsWith(item.match ?? item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMoreOpen(false)}
                className={`flex items-center gap-3 px-4 py-3 text-sm ${
                  active ? "bg-accent/15 text-foreground" : "text-muted"
                }`}
              >
                <Icon className={`h-5 w-5 shrink-0 ${active ? "text-accent" : "text-faint"}`} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </>
    )}
    <nav aria-label="Primary navigation" className="fixed inset-x-0 bottom-0 z-40 flex h-16 items-stretch border-t border-border bg-surface md:hidden">
      {primaryItems.map((item) => {
        const active = pathname.startsWith(item.match ?? item.href);
        const Icon = item.icon;
        return (
          <Link key={item.href} href={item.href} onClick={() => setMoreOpen(false)} className={`flex flex-1 flex-col items-center justify-center gap-1 px-1 text-[11px] ${active ? "text-accent" : "text-muted"}`}>
            <Icon className="h-5 w-5" />
            <span className="whitespace-nowrap">{item.label}</span>
          </Link>
        );
      })}
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setMoreOpen((open) => !open)}
        aria-expanded={moreOpen}
        aria-haspopup="dialog"
        className={`flex flex-1 flex-col items-center justify-center gap-1 px-1 text-[11px] ${
          moreOpen || overflowActive ? "text-accent" : "text-muted"
        }`}
      >
        <IconMore className="h-5 w-5" />
        <span>More</span>
      </button>
    </nav>
    </>
  );
}
