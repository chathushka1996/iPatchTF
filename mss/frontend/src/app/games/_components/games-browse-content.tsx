"use client";

import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { ActiveFilters } from "@/components/search/active-filters";
import { FilterPanel } from "@/components/search/filter-panel";
import { SortSelector } from "@/components/search/sort-selector";
import { GameGrid } from "@/components/games/game-grid";
import { Pagination } from "@/components/shared/pagination";
import { EmptyState } from "@/components/shared/empty-state";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem, GameSearchParams } from "@/types/game";
import { LayoutGrid, List, SlidersHorizontal } from "lucide-react";

export function GamesBrowseContent() {
  const searchParams = useSearchParams();
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);

  const params: GameSearchParams = {
    q: searchParams.get("q") ?? undefined,
    sort: (searchParams.get("sort") as GameSearchParams["sort"]) ?? "newest",
    page: Number(searchParams.get("page") ?? 1),
    per_page: 24,
    engine: searchParams.getAll("engine"),
    status: searchParams.getAll("status") as GameSearchParams["status"],
    genre: searchParams.getAll("genre"),
    rating: searchParams.getAll("rating") as GameSearchParams["rating"],
    author: searchParams.get("author") ?? undefined,
    has_play_online: searchParams.get("has_play_online") === "true" || undefined,
  };

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["games", params],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>("/games/search", {
        params,
      }),
    placeholderData: (prev) => prev,
  });

  const start = data ? (data.page - 1) * data.per_page + 1 : 0;
  const end = data ? Math.min(data.page * data.per_page, data.total) : 0;

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Browse Games</h1>
        <p className="mt-1 text-muted-foreground">
          Search and filter the GameVault database
        </p>
      </div>

      <div className="flex gap-8">
        <aside className="hidden w-[300px] shrink-0 lg:block">
          <FilterPanel />
        </aside>

        <Button
          variant="outline"
          size="sm"
          className="fixed bottom-4 right-4 z-40 lg:hidden"
          onClick={() => setMobileFiltersOpen(true)}
        >
          <SlidersHorizontal className="mr-2 h-4 w-4" />
          Filters
        </Button>

        {mobileFiltersOpen && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <div
              className="absolute inset-0 bg-black/50"
              onClick={() => setMobileFiltersOpen(false)}
            />
            <aside className="absolute bottom-0 left-0 right-0 max-h-[85vh] overflow-y-auto rounded-t-xl bg-background p-4">
              <FilterPanel onClose={() => setMobileFiltersOpen(false)} />
            </aside>
          </div>
        )}

        <div className="min-w-0 flex-1">
          <ActiveFilters />

          <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">
              {isLoading ? (
                <Skeleton className="inline-block h-4 w-48" />
              ) : data ? (
                `Showing ${start}–${end} of ${data.total.toLocaleString()} games`
              ) : null}
              {isFetching && !isLoading && (
                <span className="ml-2 text-indigo-500">Updating...</span>
              )}
            </p>
            <div className="flex items-center gap-2">
              <SortSelector />
              <div className="flex rounded-md border border-border">
                <Button
                  variant={viewMode === "grid" ? "secondary" : "ghost"}
                  size="icon"
                  className="h-8 w-8 rounded-r-none"
                  onClick={() => setViewMode("grid")}
                  aria-label="Grid view"
                >
                  <LayoutGrid className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "secondary" : "ghost"}
                  size="icon"
                  className="h-8 w-8 rounded-l-none"
                  onClick={() => setViewMode("list")}
                  aria-label="List view"
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>

          {!isLoading && data?.items.length === 0 ? (
            <EmptyState
              title="No games found"
              description="Try broadening your search or clearing some filters."
              actionLabel="Clear filters"
              actionHref="/games"
            />
          ) : (
            <GameGrid
              games={data?.items ?? []}
              isLoading={isLoading}
              variant={viewMode}
            />
          )}

          {data && data.pages > 1 && (
            <div className="mt-8">
              <Pagination
                page={data.page}
                totalPages={data.pages}
                basePath="/games"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
