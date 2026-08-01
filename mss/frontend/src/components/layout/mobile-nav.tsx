"use client";

import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Gamepad2, LogIn, MessageSquare, MessagesSquare, UserPlus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

import { ThemeToggle } from "./theme-toggle";

const NAV_LINKS = [
  { href: "/games", label: "Browse Games", icon: Gamepad2 },
  { href: "/community", label: "Community", icon: MessageSquare },
  { href: "/chat", label: "Chat", icon: MessagesSquare },
];

interface MobileNavProps {
  open: boolean;
  onClose: () => void;
  isAuthenticated?: boolean;
  username?: string;
  onLogout?: () => void;
}

export function MobileNav({
  open,
  onClose,
  isAuthenticated = false,
  username,
  onLogout,
}: MobileNavProps) {
  const pathname = usePathname();

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm lg:hidden"
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r bg-background shadow-xl lg:hidden"
          >
            <div className="flex items-center justify-between border-b p-4">
              <Link href="/" className="flex items-center gap-2 font-bold" onClick={onClose}>
                <Gamepad2 className="h-6 w-6 text-primary" />
                GameVault
              </Link>
              <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close menu">
                <X className="h-5 w-5" />
              </Button>
            </div>

            <nav className="flex-1 space-y-1 p-4">
              {NAV_LINKS.map((link) => {
                const Icon = link.icon;
                const isActive = pathname.startsWith(link.href);
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={onClose}
                    className={cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {link.label}
                  </Link>
                );
              })}
            </nav>

            <div className="border-t p-4">
              <div className="mb-4 flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Theme</span>
                <ThemeToggle />
              </div>
              <Separator className="mb-4" />
              {isAuthenticated ? (
                <div className="space-y-2">
                  {username && (
                    <p className="px-3 text-sm font-medium">{username}</p>
                  )}
                  <Button variant="outline" className="w-full" asChild>
                    <Link href="/profile" onClick={onClose}>Profile</Link>
                  </Button>
                  <Button variant="outline" className="w-full" asChild>
                    <Link href="/dashboard" onClick={onClose}>Dashboard</Link>
                  </Button>
                  <Button variant="outline" className="w-full" asChild>
                    <Link href="/settings" onClick={onClose}>Settings</Link>
                  </Button>
                  <Button
                    variant="destructive"
                    className="w-full"
                    onClick={() => {
                      onLogout?.();
                      onClose();
                    }}
                  >
                    Logout
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  <Button variant="outline" className="w-full" asChild>
                    <Link href="/login" onClick={onClose}>
                      <LogIn className="mr-2 h-4 w-4" />
                      Login
                    </Link>
                  </Button>
                  <Button className="w-full" asChild>
                    <Link href="/register" onClick={onClose}>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Register
                    </Link>
                  </Button>
                </div>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
