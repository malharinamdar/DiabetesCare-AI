"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Assessment" },
  { href: "/assistant", label: "Assistant" },
  { href: "/history", label: "History" },
  { href: "/method", label: "Method" },
];

export default function Header() {
  const path = usePathname();
  return (
    <header className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-5 py-6 sm:px-8">
      <Link href="/" className="flex items-center gap-2.5" aria-label="DiabetesCare home">
        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden>
          <path d="M12 2.5c3.6 4.6 6.5 8.4 6.5 12a6.5 6.5 0 0 1-13 0c0-3.6 2.9-7.4 6.5-12Z" fill="var(--color-high)" />
          <path d="M8.6 14.6h2l1.1-2.4 1.6 4.6 1.2-2.2h1" fill="none" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-[17px] font-medium tracking-tight">
          DiabetesCare<span className="text-faint"> AI</span>
        </span>
      </Link>
      <nav className="-mx-1 flex items-center gap-0.5 overflow-x-auto text-sm">
        {LINKS.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`whitespace-nowrap rounded-full px-3 py-1.5 transition-colors ${
                active ? "bg-ink text-bg" : "text-muted hover:text-ink"
              }`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
