import type { Metadata } from "next";
import Link from "next/link";
import { Plus_Jakarta_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const sans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Volleyball AI",
  description: "Offline volleyball video analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <header className="app-header">
          <div className="inner">
            <Link href="/" className="brand">
              <span className="brand-mark" aria-hidden />
              Volleyball AI
            </Link>
            <nav>
              <Link href="/">Library</Link>
              <Link href="/upload">Upload</Link>
            </nav>
          </div>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
