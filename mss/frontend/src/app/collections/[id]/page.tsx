"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { GameGrid } from "@/components/games/game-grid";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { CollectionDetail } from "@/types/collection";

interface CollectionDetailPageProps {
  params: { id: string };
}

export default function CollectionDetailPage({
  params,
}: CollectionDetailPageProps) {
  const { data: collection, isLoading } = useQuery({
    queryKey: ["collection", params.id],
    queryFn: () =>
      apiClient.get<CollectionDetail>(`/collections/${params.id}`),
  });

  if (isLoading) {
    return (
      <div className="container mx-auto max-w-7xl px-4 py-8">
        <Skeleton className="mb-4 h-10 w-64" />
        <Skeleton className="mb-2 h-4 w-96" />
        <Skeleton className="mt-8 h-64 w-full" />
      </div>
    );
  }

  if (!collection) {
    return (
      <div className="container mx-auto max-w-7xl px-4 py-8 text-center">
        <p className="text-muted-foreground">Collection not found.</p>
        <Link
          href="/collections"
          className="mt-4 inline-block text-indigo-500 hover:underline"
        >
          Back to collections
        </Link>
      </div>
    );
  }

  const games = collection.games.map((item) => item.game);

  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <nav className="mb-4 text-sm text-muted-foreground">
        <Link href="/collections" className="hover:text-foreground">
          Collections
        </Link>
        <span className="mx-2">/</span>
        <span>{collection.name}</span>
      </nav>

      <h1 className="text-3xl font-bold">{collection.name}</h1>
      {collection.description && (
        <p className="mt-2 max-w-2xl text-muted-foreground">
          {collection.description}
        </p>
      )}
      <p className="mt-2 text-sm text-muted-foreground">
        Curated by{" "}
        <Link
          href={`/profile/${collection.user.username}`}
          className="text-indigo-500 hover:underline"
        >
          {collection.user.display_name ?? collection.user.username}
        </Link>
        {" · "}
        {collection.game_count} games
      </p>

      <div className="mt-8">
        <GameGrid games={games} />
      </div>
    </div>
  );
}
