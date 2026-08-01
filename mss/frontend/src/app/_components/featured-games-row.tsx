"use client";

import { useQuery } from "@tanstack/react-query";

import { GameCard } from "@/components/games/game-card";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem } from "@/types/game";

export function FeaturedGamesRow() {
  const { data, isLoading } = useQuery({
    queryKey: ["games", "featured"],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>("/games/featured", {
        params: { per_page: 12 },
      }),
  });

  if (isLoading) {
    return (
      <div className="flex gap-4 overflow-x-auto pb-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="h-48 w-40 shrink-0 animate-pulse rounded-lg bg-muted"
          />
        ))}
      </div>
    );
  }

  return (
    <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-thin">
      {data?.items.map((game) => (
        <div key={game.id} className="w-40 shrink-0 md:w-48">
          <GameCard game={game} variant="compact" />
        </div>
      ))}
    </div>
  );
}
