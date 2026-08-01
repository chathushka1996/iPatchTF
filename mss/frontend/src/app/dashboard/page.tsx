"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { ActivityFeed } from "@/components/profile/activity-feed";
import { GameCard } from "@/components/games/game-card";
import { ReviewList } from "@/components/reviews/review-list";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { apiClient } from "@/lib/api-client";
import type { DashboardStats } from "@/types/api";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem } from "@/types/game";
import { Gamepad2, MessageSquare, Star, Bell } from "lucide-react";

export default function DashboardPage() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: () => apiClient.get<DashboardStats>("/users/me/dashboard"),
    enabled: isAuthenticated,
  });

  const { data: recentGames } = useQuery({
    queryKey: ["my-games-recent"],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>("/users/me/games", {
        params: { per_page: 4 },
      }),
    enabled: isAuthenticated,
  });

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login?redirect=/dashboard");
    }
  }, [isAuthenticated, authLoading, router]);

  if (authLoading) {
    return <DashboardSkeleton />;
  }

  if (!isAuthenticated) return null;

  const statCards = [
    {
      label: "My Games",
      value: stats?.game_count ?? 0,
      icon: Gamepad2,
      href: "/dashboard/my-games",
    },
    {
      label: "My Reviews",
      value: stats?.review_count ?? 0,
      icon: Star,
      href: "/dashboard/my-reviews",
    },
    {
      label: "Forum Posts",
      value: stats?.post_count ?? 0,
      icon: MessageSquare,
      href: "/community",
    },
    {
      label: "Notifications",
      value: stats?.unread_notifications ?? 0,
      icon: Bell,
      href: "/dashboard/notifications",
    },
  ];

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="mt-1 text-muted-foreground">
          Your personalized GameVault overview
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {statCards.map((stat) => (
          <Link key={stat.label} href={stat.href}>
            <Card className="transition-colors hover:border-indigo-500/50">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.label}
                </CardTitle>
                <stat.icon className="h-4 w-4 text-indigo-500" />
              </CardHeader>
              <CardContent>
                {statsLoading ? (
                  <Skeleton className="h-8 w-16" />
                ) : (
                  <p className="text-2xl font-bold">{stat.value}</p>
                )}
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <section>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold">Activity Feed</h2>
          </div>
          <ActivityFeed variant="dashboard" />
        </section>

        <section>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold">My Games</h2>
            <Button variant="outline" size="sm" asChild>
              <Link href="/dashboard/my-games">View all</Link>
            </Button>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {recentGames?.items.map((game) => (
              <GameCard key={game.id} game={game} variant="compact" />
            ))}
          </div>
        </section>
      </div>

      <section className="mt-8">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-semibold">Recent Reviews</h2>
          <Button variant="outline" size="sm" asChild>
            <Link href="/dashboard/my-reviews">View all</Link>
          </Button>
        </div>
        <ReviewList endpoint="/users/me/reviews" limit={5} />
      </section>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <Skeleton className="mb-8 h-10 w-48" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-lg" />
        ))}
      </div>
    </div>
  );
}
