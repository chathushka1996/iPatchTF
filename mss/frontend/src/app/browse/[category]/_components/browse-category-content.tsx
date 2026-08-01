"use client";

import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { ActiveFilters } from "@/components/search/active-filters";
import { FilterPanel } from "@/components/search/filter-panel";
import { SortSelector } from "@/components/search/sort-selector";
import { GameGrid } from "@/components/games/game-grid";
import { Pagination } from "@/components/shared/pagination";
import { EmptyState } from "@/components/shared/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem } from "@/types/game";

const CATEGORY_LABELS: Record<string, string> = {
  engine: "Engine",
  genre: "Genre",
  author: "Author",
  "adult-theme": "Adult Theme",
  transformation: "Transformation",
  multimedia: "Multimedia",
};

interface BrowseCategoryContentProps {
  category: string;
}

export function BrowseCategoryContent({ category }: BrowseCategoryContentProps) {
  const searchParams = useSearchParams();
  const value = searchParams.get("value") ?? "";
  const page = Number(searchParams.get("page") ?? 1);
  const categoryLabel = CATEGORY_LABELS[category] ?? category;

  const { data, isLoading } = useQuery({
    queryKey: ["browse", category, value, page],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>(
        `/browse/${category}/${value}`,
        { params: { page, per_page: 24 } },
      ),
    enabled: !!value,
  });

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">
          Browse by {categoryLabel}
          {value && (
            <span className="text-indigo-500">
              : {decodeURIComponent(value)}
            </span>
          )}
        </h1>
        <p className="mt-1 text-muted-foreground">
          Games filtered by {categoryLabel.toLowerCase()}
        </p>
      </div>

      <div className="flex gap-8">
        <aside className="hidden w-[300px] shrink-0 lg:block">
          <FilterPanel />
        </aside>

        <div className="min-w-0 flex-1">
          <ActiveFilters />

          <div className="mb-4 flex items-center justify-between">
            {isLoading ? (
              <Skeleton className="h-4 w-48" />
            ) : data ? (
              <p className="text-sm text-muted-foreground">
                {data.total.toLocaleString()} games found
              </p>
            ) : null}
            <SortSelector />
          </div>

          {!isLoading && data?.items.length === 0 ? (
            <EmptyState
              title="No games in this category"
              description="Try browsing a different category or clearing filters."
              actionLabel="Browse all games"
              actionHref="/games"
            />
          ) : (
            <GameGrid games={data?.items ?? []} isLoading={isLoading} />
          )}

          {data && data.pages > 1 && (
            <div className="mt-8">
              <Pagination
                page={data.page}
                totalPages={data.pages}
                basePath={`/browse/${category}?value=${encodeURIComponent(value)}`}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
