"use client";

import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import { Heart, Star } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { cn, formatNumber } from "@/lib/utils";
import type { GameListItem } from "@/types/game";

interface GameCardProps {
  game: GameListItem;
  variant?: "default" | "compact";
  className?: string;
}

const STATUS_VARIANTS: Record<
  string,
  "default" | "secondary" | "success" | "warning" | "outline" | "destructive"
> = {
  concept: "outline",
  demo: "secondary",
  alpha: "warning",
  beta: "warning",
  complete: "success",
  discontinued: "destructive",
};

export function GameCard({
  game,
  variant = "default",
  className,
}: GameCardProps) {
  const avgScore =
    typeof game.average_score === "string"
      ? parseFloat(game.average_score)
      : game.average_score;

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={cn("group", className)}
    >
      <Link
        href={`/games/${game.slug}`}
        className="block overflow-hidden rounded-lg border bg-card shadow-sm transition-shadow hover:shadow-md"
      >
        <div
          className={cn(
            "relative overflow-hidden bg-gradient-to-br from-indigo-500/20 to-purple-500/20",
            variant === "compact" ? "aspect-[4/5]" : "aspect-video",
          )}
        >
          {game.thumbnail_url ? (
            <Image
              src={game.thumbnail_url}
              alt={game.title}
              fill
              className="object-cover transition-transform group-hover:scale-105"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
            />
          ) : (
            <div className="flex h-full items-center justify-center">
              <span className="text-4xl font-bold text-primary/30">
                {game.title.charAt(0)}
              </span>
            </div>
          )}
        </div>

        <div className={cn("p-3", variant === "compact" && "p-2")}>
          <h3 className="line-clamp-2 font-semibold leading-tight group-hover:text-primary">
            {game.title}
          </h3>
          <p className="mt-1 text-sm text-muted-foreground">{game.author_name}</p>

          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Badge variant="secondary" className="text-xs">
              {game.engine_name}
            </Badge>
            <Badge
              variant={STATUS_VARIANTS[game.development_status] ?? "outline"}
              className="text-xs capitalize"
            >
              {game.development_status}
            </Badge>
          </div>

          <div className="mt-2 flex items-center gap-3 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Heart className="h-3.5 w-3.5" />
              {formatNumber(game.like_count)}
            </span>
            {avgScore > 0 && (
              <span className="flex items-center gap-1">
                <Star className="h-3.5 w-3.5 fill-warning text-warning" />
                {avgScore.toFixed(1)}
              </span>
            )}
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
