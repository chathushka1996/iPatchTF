"use client";

import Link from "next/link";
import {
  ExternalLink,
  Flag,
  Heart,
  Share2,
  Star,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Game } from "@/types/game";

interface GameDetailHeaderProps {
  game: Game;
  isLiked?: boolean;
  onLike?: () => void;
  onShare?: () => void;
  onReport?: () => void;
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

export function GameDetailHeader({
  game,
  isLiked = false,
  onLike,
  onShare,
  onReport,
  className,
}: GameDetailHeaderProps) {
  const avgScore =
    typeof game.average_score === "string"
      ? parseFloat(game.average_score)
      : game.average_score;

  const authorName = game.author.display_name || game.author.username;

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
            {game.title}
          </h1>
          <p className="text-muted-foreground">
            by{" "}
            <Link
              href={`/users/${game.author.username}`}
              className="font-medium text-primary hover:underline"
            >
              {authorName}
            </Link>
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">{game.engine.name}</Badge>
            <Badge
              variant={STATUS_VARIANTS[game.development_status] ?? "outline"}
              className="capitalize"
            >
              {game.development_status}
            </Badge>
            {avgScore > 0 && (
              <Badge variant="default" className="gap-1">
                <Star className="h-3 w-3 fill-current" />
                {avgScore.toFixed(1)} ({game.review_count} reviews)
              </Badge>
            )}
            <Badge variant="outline">{game.rating}</Badge>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            variant={isLiked ? "default" : "outline"}
            size="sm"
            onClick={onLike}
          >
            <Heart
              className={cn("mr-2 h-4 w-4", isLiked && "fill-current")}
            />
            {game.like_count}
          </Button>
          {game.play_online_url && (
            <Button size="sm" asChild>
              <a
                href={game.play_online_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="mr-2 h-4 w-4" />
                Play Online
              </a>
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={onShare}>
            <Share2 className="mr-2 h-4 w-4" />
            Share
          </Button>
          <Button variant="ghost" size="sm" onClick={onReport}>
            <Flag className="mr-2 h-4 w-4" />
            Report
          </Button>
        </div>
      </div>
    </div>
  );
}
