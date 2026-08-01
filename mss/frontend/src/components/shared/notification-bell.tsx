"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { Bell } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { isAuthenticated } from "@/lib/auth";
import { get as apiGet, patch } from "@/lib/api-client";
import { formatRelativeDate } from "@/lib/utils";
import type { Notification } from "@/types/notification";

interface UnreadCountResponse {
  count: number;
}

export function NotificationBell() {
  const [authed, setAuthed] = useState(false);

  useEffect(() => {
    setAuthed(isAuthenticated());
  }, []);

  const { data: unreadData } = useQuery({
    queryKey: ["notifications", "unread-count"],
    queryFn: () => apiGet<UnreadCountResponse>("/v1/notifications/unread-count"),
    enabled: authed,
    refetchInterval: 60_000,
  });

  const { data: notifications, refetch } = useQuery({
    queryKey: ["notifications", "recent"],
    queryFn: () =>
      apiGet<{ items: Notification[] }>("/v1/notifications", {
        params: { per_page: 10 },
      }),
    enabled: authed,
  });

  const unreadCount = unreadData?.count ?? 0;

  const markAllRead = async () => {
    await patch("/v1/notifications/read", {});
    refetch();
  };

  if (!authed) return null;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="relative" aria-label="Notifications">
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <Badge
              variant="destructive"
              className="absolute -right-1 -top-1 flex h-5 min-w-5 items-center justify-center rounded-full px-1 text-[10px]"
            >
              {unreadCount > 99 ? "99+" : unreadCount}
            </Badge>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        <DropdownMenuLabel className="flex items-center justify-between">
          Notifications
          {unreadCount > 0 && (
            <button
              type="button"
              onClick={markAllRead}
              className="text-xs font-normal text-primary hover:underline"
            >
              Mark all read
            </button>
          )}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <ScrollArea className="h-72">
          {(notifications?.items ?? []).length === 0 ? (
            <p className="p-4 text-center text-sm text-muted-foreground">
              No notifications
            </p>
          ) : (
            (notifications?.items ?? []).map((notification) => (
              <DropdownMenuItem key={notification.id} asChild>
                <Link
                  href={notification.link ?? "#"}
                  className={`flex flex-col items-start gap-1 p-3 ${
                    !notification.is_read ? "bg-primary/5" : ""
                  }`}
                >
                  <span className="text-sm font-medium">
                    {notification.title}
                  </span>
                  {notification.body && (
                    <span className="line-clamp-2 text-xs text-muted-foreground">
                      {notification.body}
                    </span>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {formatRelativeDate(notification.created_at)}
                  </span>
                </Link>
              </DropdownMenuItem>
            ))
          )}
        </ScrollArea>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
