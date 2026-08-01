"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { ReviewList } from "@/components/reviews/review-list";
import { EmptyState } from "@/components/shared/empty-state";
import { Pagination } from "@/components/shared/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { Review } from "@/types/review";

export default function MyReviewsPage() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();

  const { data, isLoading } = useQuery({
    queryKey: ["my-reviews"],
    queryFn: () =>
      apiClient.get<PaginatedResponse<Review>>("/users/me/reviews", {
        params: { per_page: 20 },
      }),
    enabled: isAuthenticated,
  });

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login?redirect=/dashboard/my-reviews");
    }
  }, [isAuthenticated, authLoading, router]);

  if (authLoading || isLoading) {
    return (
      <div className="container mx-auto max-w-3xl px-4 py-8">
        <Skeleton className="mb-6 h-10 w-48" />
        <div className="space-y-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return null;

  return (
    <div className="container mx-auto max-w-3xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">My Reviews</h1>
        <p className="mt-1 text-muted-foreground">
          Reviews you&apos;ve written on GameVault
        </p>
      </div>

      {data?.items.length === 0 ? (
        <EmptyState
          title="No reviews yet"
          description="Browse games and share your thoughts with the community."
          actionLabel="Browse Games"
          actionHref="/games"
        />
      ) : (
        <ReviewList
          reviews={data?.items}
          showGameLink
        />
      )}

      {data && data.pages > 1 && (
        <div className="mt-8">
          <Pagination
            page={data.page}
            totalPages={data.pages}
            basePath="/dashboard/my-reviews"
          />
        </div>
      )}
    </div>
  );
}
