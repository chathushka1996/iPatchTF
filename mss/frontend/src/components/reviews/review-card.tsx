"use client";

import Link from "next/link";
import { ThumbsDown, ThumbsUp } from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { formatRelativeDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { Review } from "@/types/review";

interface ReviewCardProps {
  review: Review;
  variant?: "full" | "excerpt";
  onVote?: (reviewId: string, isHelpful: boolean) => void;
  className?: string;
}

function getScoreVariant(score: number): "destructive" | "warning" | "success" | "default" {
  if (score <= 3) return "destructive";
  if (score <= 6) return "warning";
  if (score <= 8) return "default";
  return "success";
}

export function ReviewCard({
  review,
  variant = "full",
  onVote,
  className,
}: ReviewCardProps) {
  const displayName =
    review.user.display_name || review.user.username;

  return (
    <Card className={cn(className)}>
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <Link href={`/users/${review.user.username}`}>
            <Avatar className="h-10 w-10">
              <AvatarImage src={review.user.avatar_url ?? undefined} />
              <AvatarFallback name={displayName} />
            </Avatar>
          </Link>

          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Link
                href={`/users/${review.user.username}`}
                className="font-medium hover:text-primary"
              >
                {displayName}
              </Link>
              <Badge variant={getScoreVariant(review.score)}>
                {review.score}/10
              </Badge>
              {review.version_reviewed && (
                <span className="text-xs text-muted-foreground">
                  v{review.version_reviewed}
                </span>
              )}
              <span className="text-xs text-muted-foreground">
                {formatRelativeDate(review.created_at)}
              </span>
              {review.is_edited && (
                <span className="text-xs text-muted-foreground">(edited)</span>
              )}
            </div>

            <p
              className={cn(
                "text-sm leading-relaxed",
                variant === "excerpt" && "line-clamp-3",
              )}
            >
              {review.body}
            </p>

            {variant === "full" && onVote && (
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onVote(review.id, true)}
                >
                  <ThumbsUp className="mr-1 h-3.5 w-3.5" />
                  {review.helpful_count}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onVote(review.id, false)}
                >
                  <ThumbsDown className="mr-1 h-3.5 w-3.5" />
                  {review.not_helpful_count}
                </Button>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
