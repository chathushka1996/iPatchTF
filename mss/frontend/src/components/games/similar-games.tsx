"use client";

import { GameCard } from "@/components/games/game-card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { GameListItem } from "@/types/game";

interface SimilarGamesProps {
  games: GameListItem[];
  isLoading?: boolean;
  title?: string;
  className?: string;
}

export function SimilarGames({
  games,
  isLoading = false,
  title = "Similar Games",
  className,
}: SimilarGamesProps) {
  if (isLoading) {
    return (
      <div className={cn("space-y-4", className)}>
        <h2 className="text-xl font-semibold">{title}</h2>
        <div className="flex gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-40 shrink-0 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (games.length === 0) return null;

  return (
    <div className={cn("space-y-4", className)}>
      <h2 className="text-xl font-semibold">{title}</h2>
      <ScrollArea className="w-full whitespace-nowrap">
        <div className="flex gap-4 pb-4">
          {games.map((game) => (
            <div key={game.id} className="w-44 shrink-0 md:w-52">
              <GameCard game={game} variant="compact" />
            </div>
          ))}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
