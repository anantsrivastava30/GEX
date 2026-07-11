"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ComponentType } from "react";
import {
  IconAI,
  IconCalendar,
  IconCongress,
  IconFlow,
  IconMarket,
  IconNews,
  IconScreener,
  IconTicker,
  IconTrackRecord,
  IconTools,
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
    ],
  },
  {
    heading: "Research",
    items: [
      { href: "/news", label: "News", icon: IconNews },
      { href: "/track-record", label: "Track Record", icon: IconTrackRecord },
      { href: "/ai", label: "AI Analysis", icon: IconAI },
      { href: "/calendar", label: "Calendar", icon: IconCalendar },
      { href: "/congress", label: "Congress", icon: IconCongress, soon: true },
      { href: "/tools/binomial", label: "Tools", icon: IconTools },
    ],
  },
];

export default function SidebarNav() {
  const pathname = usePathname();
  return (
    <aside className="fixed inset-y-0 left-0 z-20 flex w-52 flex-col border-r border-border bg-surface">
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
  );
}
