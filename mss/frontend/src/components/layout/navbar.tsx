"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Gamepad2,
  LayoutDashboard,
  LogOut,
  Menu,
  Search,
  Settings,
  User,
} from "lucide-react";

import { NotificationBell } from "@/components/shared/notification-bell";
import { SearchBar } from "@/components/search/search-bar";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { clearTokens, getAccessToken, isAuthenticated } from "@/lib/auth";
import { get } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import type { UserBrief } from "@/types/user";

import { MobileNav } from "./mobile-nav";
import { ThemeToggle } from "./theme-toggle";

const NAV_LINKS = [
  { href: "/games", label: "Browse Games" },
  { href: "/community", label: "Community" },
  { href: "/chat", label: "Chat" },
];

export function Navbar() {
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [user, setUser] = useState<UserBrief | null>(null);
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    const authed = isAuthenticated();
    setAuthenticated(authed);

    if (authed) {
      get<UserBrief>("/v1/users/me")
        .then(setUser)
        .catch(() => setUser(null));
    }

    const handleLogout = () => {
      setUser(null);
      setAuthenticated(false);
    };

    window.addEventListener("auth:logout", handleLogout);
    return () => window.removeEventListener("auth:logout", handleLogout);
  }, []);

  const handleLogout = () => {
    clearTokens();
    setUser(null);
    setAuthenticated(false);
    router.push("/");
    router.refresh();
  };

  const displayName = user?.display_name || user?.username;

  return (
    <>
      <header className="sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur-md">
        <div className="container mx-auto flex h-16 max-w-7xl items-center gap-4 px-4">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </Button>

          <Link href="/" className="flex shrink-0 items-center gap-2 font-bold">
            <Gamepad2 className="h-6 w-6 text-primary" />
            <span className="hidden sm:inline">GameVault</span>
          </Link>

          <nav className="hidden items-center gap-1 lg:flex">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                {link.label}
              </Link>
            ))}
          </nav>

          <div className="flex flex-1 items-center justify-end gap-2">
            <div
              className={cn(
                "hidden max-w-sm flex-1 md:block",
                searchOpen && "block absolute inset-x-16 top-3 md:static md:inset-auto",
              )}
            >
              <SearchBar placeholder="Search games..." className="w-full" />
            </div>

            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setSearchOpen(!searchOpen)}
              aria-label="Toggle search"
            >
              <Search className="h-5 w-5" />
            </Button>

            <ThemeToggle />

            {authenticated && <NotificationBell />}

            {authenticated && user ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    className="relative h-9 gap-2 px-2"
                  >
                    <Avatar className="h-8 w-8">
                      <AvatarImage src={user.avatar_url ?? undefined} />
                      <AvatarFallback name={displayName ?? user.username} />
                    </Avatar>
                    <span className="hidden text-sm font-medium sm:inline">
                      {displayName}
                    </span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-48">
                  <DropdownMenuLabel>
                    <div className="flex flex-col">
                      <span>{displayName}</span>
                      <span className="text-xs font-normal text-muted-foreground">
                        @{user.username}
                      </span>
                    </div>
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem asChild>
                    <Link href={`/users/${user.username}`}>
                      <User className="mr-2 h-4 w-4" />
                      Profile
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/dashboard">
                      <LayoutDashboard className="mr-2 h-4 w-4" />
                      Dashboard
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/settings">
                      <Settings className="mr-2 h-4 w-4" />
                      Settings
                    </Link>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleLogout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <div className="hidden items-center gap-2 sm:flex">
                <Button variant="ghost" asChild>
                  <Link href="/login">Login</Link>
                </Button>
                <Button asChild>
                  <Link href="/register">Register</Link>
                </Button>
              </div>
            )}
          </div>
        </div>
      </header>

      <MobileNav
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        isAuthenticated={authenticated}
        username={displayName ?? undefined}
        onLogout={handleLogout}
      />
    </>
  );
}
