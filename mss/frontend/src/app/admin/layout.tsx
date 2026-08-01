"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";

import { Sidebar } from "@/components/layout/sidebar";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import {
  BarChart3,
  Flag,
  Gamepad2,
  LayoutDashboard,
  Users,
} from "lucide-react";

const adminNavItems = [
  { href: "/admin", label: "Dashboard", icon: LayoutDashboard },
  { href: "/admin/users", label: "Users", icon: Users },
  { href: "/admin/games", label: "Games", icon: Gamepad2 },
  { href: "/admin/reports", label: "Reports", icon: Flag },
  { href: "/admin/analytics", label: "Analytics", icon: BarChart3 },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { user, isAuthenticated, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!isLoading) {
      if (!isAuthenticated) {
        router.push("/login?redirect=/admin");
      } else if (user?.role !== "admin") {
        router.push("/");
      }
    }
  }, [isAuthenticated, isLoading, user, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)]">
        <Skeleton className="hidden w-56 shrink-0 md:block" />
        <div className="flex-1 p-8">
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  if (!isAuthenticated || user?.role !== "admin") {
    return null;
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)]">
      <Sidebar
        items={adminNavItems.map((item) => ({
          ...item,
          active: pathname === item.href,
        }))}
        title="Admin"
        className="hidden w-56 shrink-0 border-r border-border md:flex"
      />
      <div className="flex-1 overflow-auto p-4 md:p-8">{children}</div>
    </div>
  );
}
