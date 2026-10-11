import type { Metadata } from "next";
import { Geist, Geist_Mono, Newsreader } from "next/font/google";
import Header from "@/components/Header";
import "./globals.css";

const newsreader = Newsreader({ subsets: ["latin"], variable: "--font-newsreader", style: ["normal", "italic"] });
const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: { default: "DiabetesCare — diabetes risk screening", template: "%s · DiabetesCare" },
  description:
    "Estimate type 2 diabetes risk from eight routine measurements, see what drives the estimate, and get personal guidance.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${newsreader.variable} ${geist.variable} ${geistMono.variable}`}>
      <body className="min-h-dvh">
        <Header />
        <main className="mx-auto w-full max-w-6xl px-5 sm:px-8">{children}</main>
        <footer className="mx-auto mt-24 w-full max-w-6xl px-5 pb-10 sm:px-8">
          <div className="flex flex-col gap-3 border-t border-line pt-6 text-sm text-muted sm:flex-row sm:items-center sm:justify-between">
            <p>
              DiabetesCare by{" "}
              <a className="text-ink underline decoration-line underline-offset-4 hover:decoration-ink" href="https://github.com/malharinamdar">
                Malhar Inamdar
              </a>
              <span className="mx-2 text-faint">·</span>
              <span className="text-faint">A screening estimate, not a diagnosis.</span>
            </p>
            <p className="num text-xs">
              <a className="hover:text-ink" href="https://github.com/malharinamdar/DiabetesCare-AI">source</a>
              <span className="mx-2 text-faint">/</span>
              <a className="hover:text-ink" href="https://github.com/malharinamdar/DiabetesCare-AI/tree/legacy-streamlit">
                v1 (streamlit)
              </a>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
