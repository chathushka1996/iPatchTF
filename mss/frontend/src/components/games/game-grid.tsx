import { GameCard } from "@/components/games/game-card";
import { EmptyState } from "@/components/shared/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { GameListItem } from "@/types/game";

interface GameGridProps {
  games: GameListItem[];
  isLoading?: boolean;
  className?: string;
  emptyMessage?: string;
}

export function GameGrid({
  games,
  isLoading = false,
  className,
  emptyMessage = "No games found.",
}: GameGridProps) {
  if (isLoading) {
    return (
      <div
        className={cn(
          "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
          className,
        )}
      >
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="space-y-3">
            <Skeleton className="aspect-video w-full rounded-lg" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        ))}
      </div>
    );
  }

  if (games.length === 0) {
    return (
      <EmptyState
        title="No games"
        message={emptyMessage}
        icon="gamepad"
      />
    );
  }

  return (
    <div
      className={cn(
        "grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
        className,
      )}
    >
      {games.map((game) => (
        <GameCard key={game.id} game={game} />
      ))}
    </div>
  );
}
