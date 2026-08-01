"use client";

import { useQuery } from "@tanstack/react-query";

import { GameGrid } from "@/components/games/game-grid";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem } from "@/types/game";

interface HomepageGameSectionProps {
  endpoint: string;
  sort?: string;
  limit?: number;
}

export function HomepageGameSection({
  endpoint,
  sort,
  limit = 12,
}: HomepageGameSectionProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["homepage-games", endpoint, sort, limit],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>(endpoint, {
        params: { sort, per_page: limit },
      }),
  });

  return <GameGrid games={data?.items ?? []} isLoading={isLoading} />;
}
