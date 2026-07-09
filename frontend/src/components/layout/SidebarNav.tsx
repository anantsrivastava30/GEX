"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/market", label: "Market", icon: "◫" },
  { href: "/flow", label: "Flow", icon: "≋" },
  { href: "/stock/SPY", label: "Tickers", icon: "◈", match: "/stock" },
  { href: "/screener", label: "Screener", icon: "⌕", soon: true },
  { href: "/calendar", label: "Calendar", icon: "▤", soon: true },
  { href: "/congress", label: "Congress", icon: "⚖", soon: true },
  { href: "/news", label: "News", icon: "☰" },
  { href: "/ai", label: "AI Analysis", icon: "✦", soon: true },
  { href: "/tools/binomial", label: "Tools", icon: "⌥", soon: true },
];

export default function SidebarNav() {
  const pathname = usePathname();
  return (
    <aside className="fixed inset-y-0 left-0 z-20 flex w-52 flex-col border-r border-border bg-surface">
      <Link href="/market" className="flex items-center gap-2 px-4 py-4">
        <span className="text-xl">🐋</span>
        <span className="font-semibold tracking-tight">GEX Terminal</span>
      </Link>
      <nav className="flex-1 space-y-0.5 px-2 py-2">
        {NAV.map((item) => {
          const active = pathname.startsWith(item.match ?? item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                active
                  ? "bg-accent/15 text-foreground"
                  : "text-muted hover:bg-surface-hover hover:text-foreground"
              }`}
            >
              <span className="w-4 text-center">{item.icon}</span>
              {item.label}
              {item.soon && (
                <span className="ml-auto rounded bg-border px-1.5 py-0.5 text-[10px] uppercase text-muted">
                  soon
                </span>
              )}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-border px-4 py-3 text-[11px] leading-4 text-muted">
        Research &amp; education only. Not financial advice.
      </div>
    </aside>
  );
}
