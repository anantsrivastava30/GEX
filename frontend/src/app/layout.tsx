import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import SidebarNav from "@/components/layout/SidebarNav";
import TopBar from "@/components/layout/TopBar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GEX Terminal — Dealer Positioning, Explained",
  description:
    "Options analytics: gamma exposure, gamma-gap signals, IV skew, unusual activity, and market context.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full">
        <SidebarNav />
        <div className="pl-52">
          <TopBar />
          <main className="px-6 py-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
