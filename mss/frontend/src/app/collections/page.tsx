"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { EmptyState } from "@/components/shared/empty-state";
import { Pagination } from "@/components/shared/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { Collection } from "@/types/collection";

export default function CollectionsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["collections", "public"],
    queryFn: () =>
      apiClient.get<PaginatedResponse<Collection>>("/collections", {
        params: { per_page: 24 },
      }),
  });

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Collections</h1>
        <p className="mt-1 text-muted-foreground">
          Curated game lists from the community
        </p>
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-40 rounded-lg" />
          ))}
        </div>
      ) : data?.items.length === 0 ? (
        <EmptyState
          title="No collections yet"
          description="Public collections will appear here as users create them."
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {data?.items.map((collection) => (
            <Link
              key={collection.id}
              href={`/collections/${collection.id}`}
              className="group rounded-lg border border-border p-4 transition-colors hover:border-indigo-500/50"
            >
              <h2 className="font-semibold group-hover:text-indigo-500">
                {collection.name}
              </h2>
              {collection.description && (
                <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                  {collection.description}
                </p>
              )}
              <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                <span>by {collection.user.display_name ?? collection.user.username}</span>
                <span>{collection.game_count} games</span>
              </div>
            </Link>
          ))}
        </div>
      )}

      {data && data.pages > 1 && (
        <div className="mt-8">
          <Pagination
            page={data.page}
            totalPages={data.pages}
            basePath="/collections"
          />
        </div>
      )}
    </div>
  );
}
