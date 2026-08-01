"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { ReviewCard } from "@/components/reviews/review-card";
import { EmptyState } from "@/components/shared/empty-state";
import { Pagination } from "@/components/shared/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { get } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import type { PaginatedResponse } from "@/types/api";
import type { Review } from "@/types/review";

type ReviewSort = "newest" | "oldest" | "highest" | "lowest" | "helpful";

const SORT_OPTIONS: { value: ReviewSort; label: string }[] = [
  { value: "newest", label: "Newest" },
  { value: "oldest", label: "Oldest" },
  { value: "highest", label: "Highest Rated" },
  { value: "lowest", label: "Lowest Rated" },
  { value: "helpful", label: "Most Helpful" },
];

interface ReviewListProps {
  gameId?: string;
  endpoint?: string;
  limit?: number;
  variant?: "full" | "excerpt";
  onVote?: (reviewId: string, isHelpful: boolean) => void;
  className?: string;
}

export function ReviewList({
  gameId,
  endpoint,
  limit,
  variant = "full",
  onVote,
  className,
}: ReviewListProps) {
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState<ReviewSort>("newest");

  const url = endpoint ?? (gameId ? `/v1/games/${gameId}/reviews` : null);

  const { data, isLoading } = useQuery({
    queryKey: ["reviews", url, page, sort, limit],
    queryFn: () =>
      get<PaginatedResponse<Review>>(url!, {
        params: { page, sort, per_page: limit ?? 10 },
      }),
    enabled: Boolean(url),
  });

  if (isLoading) {
    return (
      <div className={cn("space-y-4", className)}>
        {Array.from({ length: limit ?? 5 }).map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-lg" />
        ))}
      </div>
    );
  }

  const reviews = data?.items ?? [];

  if (reviews.length === 0) {
    return (
      <EmptyState
        title="No reviews yet"
        message="Be the first to review this game!"
        icon="star"
      />
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {variant === "full" && (
        <div className="flex justify-end">
          <Select value={sort} onValueChange={(v) => setSort(v as ReviewSort)}>
            <SelectTrigger className="w-[160px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SORT_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <div className="space-y-3">
        {reviews.map((review) => (
          <ReviewCard
            key={review.id}
            review={review}
            variant={variant}
            onVote={onVote}
          />
        ))}
      </div>

      {variant === "full" && data && data.pages > 1 && (
        <Pagination
          page={data.page}
          totalPages={data.pages}
          onPageChange={setPage}
        />
      )}
    </div>
  );
}
