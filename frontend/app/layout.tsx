import './globals.css';
import Link from 'next/link';
import type { ReactNode } from 'react';

export const metadata = { title: 'Tracker Lab', description: 'Multi-object tracking research playground' };

const NAV = [
  { href: '/tracker-lab', label: 'Tracker Lab' },
  { href: '/compare', label: 'Compare' },
  { href: '/algorithm', label: 'Algorithm' },
  { href: '/stress-test', label: 'Stress Test' },
];

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans">
        <header className="border-b border-neutral-800 px-6 py-4 flex gap-6 items-center">
          <Link href="/" className="font-semibold tracking-tight">Tracker Lab</Link>
          <nav className="flex gap-4 text-sm text-neutral-400">
            {NAV.map((n) => (
              <Link key={n.href} href={n.href} className="hover:text-white">{n.label}</Link>
            ))}
          </nav>
        </header>
        <main className="px-6 py-8 max-w-6xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
