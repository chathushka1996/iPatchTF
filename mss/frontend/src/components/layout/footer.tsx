import Link from "next/link";
import { Gamepad2, Github } from "lucide-react";

import { Separator } from "@/components/ui/separator";

const FOOTER_COLUMNS = [
  {
    title: "About",
    links: [
      { href: "/about", label: "About GameVault" },
      { href: "/faq", label: "FAQ" },
      { href: "/contact", label: "Contact" },
    ],
  },
  {
    title: "Browse",
    links: [
      { href: "/games?filter=engine", label: "By Engine" },
      { href: "/games?filter=genre", label: "By Genre" },
      { href: "/games?filter=status", label: "By Status" },
    ],
  },
  {
    title: "Community",
    links: [
      { href: "/community", label: "Forums" },
      { href: "/chat", label: "Chat" },
    ],
  },
  {
    title: "Links",
    links: [
      { href: "/api/docs", label: "API Docs" },
      {
        href: "https://github.com",
        label: "GitHub",
        external: true,
      },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t bg-surface">
      <div className="container mx-auto max-w-7xl px-4 py-12">
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-5">
          <div className="lg:col-span-1">
            <Link href="/" className="flex items-center gap-2 font-bold">
              <Gamepad2 className="h-6 w-6 text-primary" />
              GameVault
            </Link>
            <p className="mt-3 text-sm text-muted-foreground">
              Discover, share, and discuss interactive games with a passionate
              community.
            </p>
          </div>

          {FOOTER_COLUMNS.map((column) => (
            <div key={column.title}>
              <h3 className="mb-3 text-sm font-semibold">{column.title}</h3>
              <ul className="space-y-2">
                {column.links.map((link) => (
                  <li key={link.href}>
                    {link.external ? (
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
                      >
                        {link.label === "GitHub" && (
                          <Github className="h-3.5 w-3.5" />
                        )}
                        {link.label}
                      </a>
                    ) : (
                      <Link
                        href={link.href}
                        className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                      >
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <Separator className="my-8" />

        <div className="flex flex-col items-center justify-between gap-4 text-sm text-muted-foreground sm:flex-row">
          <p>&copy; {new Date().getFullYear()} GameVault. All rights reserved.</p>
          <div className="flex gap-4">
            <Link href="/privacy" className="hover:text-foreground">
              Privacy
            </Link>
            <Link href="/terms" className="hover:text-foreground">
              Terms
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
