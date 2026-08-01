"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { EmptyState } from "@/components/shared/empty-state";
import { Pagination } from "@/components/shared/pagination";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem } from "@/types/game";
import { BarChart3, Edit, Plus } from "lucide-react";

export default function MyGamesPage() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();

  const { data, isLoading } = useQuery({
    queryKey: ["my-games"],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>("/users/me/games", {
        params: { per_page: 20 },
      }),
    enabled: isAuthenticated,
  });

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login?redirect=/dashboard/my-games");
    }
  }, [isAuthenticated, authLoading, router]);

  if (authLoading || isLoading) {
    return (
      <div className="container mx-auto max-w-5xl px-4 py-8">
        <Skeleton className="mb-6 h-10 w-48" />
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return null;

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">My Games</h1>
          <p className="mt-1 text-muted-foreground">
            Manage your submitted games
          </p>
        </div>
        <Button asChild>
          <Link href="/games/submit">
            <Plus className="mr-2 h-4 w-4" />
            Submit Game
          </Link>
        </Button>
      </div>

      {data?.items.length === 0 ? (
        <EmptyState
          title="No games submitted yet"
          description="Share your first game with the GameVault community."
          actionLabel="Submit a Game"
          actionHref="/games/submit"
        />
      ) : (
        <div className="space-y-3">
          {data?.items.map((game) => (
            <div
              key={game.id}
              className="flex items-center gap-4 rounded-lg border border-border p-4"
            >
              {game.thumbnail_url && (
                <img
                  src={game.thumbnail_url}
                  alt=""
                  className="h-16 w-16 rounded-md object-cover"
                />
              )}
              <div className="min-w-0 flex-1">
                <Link
                  href={`/games/${game.slug}`}
                  className="font-semibold hover:text-indigo-500"
                >
                  {game.title}
                </Link>
                <div className="mt-1 flex flex-wrap gap-2">
                  <Badge variant="secondary">{game.engine_name}</Badge>
                  <Badge variant="outline">{game.development_status}</Badge>
                </div>
                <div className="mt-1 flex gap-4 text-xs text-muted-foreground">
                  <span>{game.like_count} likes</span>
                  <span>{game.review_count} reviews</span>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" asChild>
                  <Link href={`/games/${game.slug}/edit`}>
                    <Edit className="mr-1 h-3 w-3" />
                    Edit
                  </Link>
                </Button>
                <Button variant="ghost" size="sm" asChild>
                  <Link href={`/games/${game.slug}`}>
                    <BarChart3 className="h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {data && data.pages > 1 && (
        <div className="mt-8">
          <Pagination
            page={data.page}
            totalPages={data.pages}
            basePath="/dashboard/my-games"
          />
        </div>
      )}
    </div>
  );
}
