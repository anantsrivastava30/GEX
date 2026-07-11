// Inline feather-style icons (dependency-free, currentColor stroke) so the
// sidebar and chrome read like a real product instead of unicode glyphs.

type IconProps = { className?: string };

function base(children: React.ReactNode) {
  return function Icon({ className }: IconProps) {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
        aria-hidden
      >
        {children}
      </svg>
    );
  };
}

export const IconMarket = base(
  <>
    <path d="M3 3v18h18" />
    <path d="M7 15l3-3 3 2 5-6" />
  </>,
);

export const IconFlow = base(
  <>
    <path d="M4 7h10" />
    <path d="M4 12h16" />
    <path d="M4 17h7" />
    <circle cx="18" cy="7" r="2" />
    <circle cx="15" cy="17" r="2" />
  </>,
);

export const IconTicker = base(
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M7 9h4M7 13h10M7 16h6" />
  </>,
);

export const IconScreener = base(
  <>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3" />
  </>,
);

export const IconCalendar = base(
  <>
    <rect x="3" y="4" width="18" height="17" rx="2" />
    <path d="M3 9h18M8 2v4M16 2v4" />
  </>,
);

export const IconCongress = base(
  <>
    <path d="M12 3l9 5H3l9-5z" />
    <path d="M5 8v9M10 8v9M14 8v9M19 8v9M3 21h18" />
  </>,
);

export const IconNews = base(
  <>
    <path d="M4 5h13v14H6a2 2 0 01-2-2V5z" />
    <path d="M17 8h3v9a2 2 0 01-2 2M8 9h5M8 13h5" />
  </>,
);

export const IconAI = base(
  <>
    <path d="M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8L12 3z" />
    <path d="M18 15l.9 2.1L21 18l-2.1.9L18 21l-.9-2.1L15 18l2.1-.9L18 15z" />
  </>,
);

export const IconTrackRecord = base(
  <>
    <path d="M4 19V5M4 19h16" />
    <path d="M7 15l4-4 3 2 4-6" />
    <circle cx="7" cy="15" r="1" />
    <circle cx="11" cy="11" r="1" />
    <circle cx="14" cy="13" r="1" />
    <circle cx="18" cy="7" r="1" />
  </>,
);

export const IconTools = base(
  <>
    <path d="M4 20l6-6" />
    <path d="M14 6l4-4 4 4-4 4" />
    <path d="M14 10l-8 8-2 2" />
    <path d="M15 5l4 4" />
  </>,
);

export const IconSearch = base(
  <>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3" />
  </>,
);
