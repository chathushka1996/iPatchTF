"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { EmptyState } from "@/components/shared/empty-state";
import { Pagination } from "@/components/shared/pagination";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { Notification } from "@/types/notification";
import { formatDistanceToNow } from "date-fns";
import { Bell, CheckCheck } from "lucide-react";

export default function NotificationsPage() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["notifications"],
    queryFn: () =>
      apiClient.get<PaginatedResponse<Notification>>(
        "/users/me/notifications",
        { params: { per_page: 30 } },
      ),
    enabled: isAuthenticated,
  });

  const markAllRead = useMutation({
    mutationFn: () => apiClient.post("/users/me/notifications/read-all"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  const markRead = useMutation({
    mutationFn: (id: string) =>
      apiClient.post(`/users/me/notifications/${id}/read`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login?redirect=/dashboard/notifications");
    }
  }, [isAuthenticated, authLoading, router]);

  if (authLoading || isLoading) {
    return (
      <div className="container mx-auto max-w-2xl px-4 py-8">
        <Skeleton className="mb-6 h-10 w-48" />
        <div className="space-y-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return null;

  return (
    <div className="container mx-auto max-w-2xl px-4 py-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Notifications</h1>
          <p className="mt-1 text-muted-foreground">
            Stay up to date with your GameVault activity
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => markAllRead.mutate()}
          disabled={markAllRead.isPending}
        >
          <CheckCheck className="mr-2 h-4 w-4" />
          Mark all read
        </Button>
      </div>

      {data?.items.length === 0 ? (
        <EmptyState
          title="No notifications"
          description="You're all caught up! Notifications will appear here."
          icon={Bell}
        />
      ) : (
        <ul className="divide-y divide-border rounded-lg border border-border">
          {data?.items.map((notification) => (
            <li
              key={notification.id}
              className={`flex gap-3 p-4 transition-colors ${
                !notification.is_read ? "bg-indigo-500/5" : ""
              }`}
            >
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">{notification.title}</p>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  {notification.body}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {formatDistanceToNow(new Date(notification.created_at), {
                    addSuffix: true,
                  })}
                </p>
              </div>
              {!notification.is_read && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => markRead.mutate(notification.id)}
                >
                  Mark read
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}

      {data && data.pages > 1 && (
        <div className="mt-8">
          <Pagination
            page={data.page}
            totalPages={data.pages}
            basePath="/dashboard/notifications"
          />
        </div>
      )}
    </div>
  );
}
