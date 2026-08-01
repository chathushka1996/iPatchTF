import {
  Gamepad2,
  Inbox,
  MessageSquare,
  Search,
  Star,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type EmptyStateIcon = "inbox" | "gamepad" | "search" | "star" | "message";

const ICONS: Record<
  EmptyStateIcon,
  React.ComponentType<{ className?: string }>
> = {
  inbox: Inbox,
  gamepad: Gamepad2,
  search: Search,
  star: Star,
  message: MessageSquare,
};

interface EmptyStateProps {
  title: string;
  message?: string;
  icon?: EmptyStateIcon;
  actionLabel?: string;
  onAction?: () => void;
  actionHref?: string;
  className?: string;
}

export function EmptyState({
  title,
  message,
  icon = "inbox",
  actionLabel,
  onAction,
  className,
}: EmptyStateProps) {
  const Icon = ICONS[icon];

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-lg border border-dashed p-12 text-center",
        className,
      )}
    >
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
        <Icon className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
      {message && (
        <p className="mt-2 max-w-sm text-sm text-muted-foreground">{message}</p>
      )}
      {actionLabel && onAction && (
        <Button className="mt-6" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
}
