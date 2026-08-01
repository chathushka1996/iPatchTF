"use client";

import { cn } from "@/lib/utils";

interface SidebarProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  sticky?: boolean;
}

export function Sidebar({
  children,
  className,
  title,
  sticky = true,
}: SidebarProps) {
  return (
    <aside
      className={cn(
        "w-full shrink-0 lg:w-64",
        sticky && "lg:sticky lg:top-20 lg:self-start",
        className,
      )}
    >
      {title && (
        <h2 className="mb-4 text-lg font-semibold">{title}</h2>
      )}
      <div className="rounded-lg border bg-card p-4">{children}</div>
    </aside>
  );
}

interface SidebarSectionProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function SidebarSection({
  title,
  children,
  className,
}: SidebarSectionProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
      {children}
    </div>
  );
}
