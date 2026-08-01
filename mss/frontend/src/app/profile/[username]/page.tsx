"use client";

import { useQuery } from "@tanstack/react-query";

import { ProfileHeader } from "@/components/profile/profile-header";
import { UserStats } from "@/components/profile/user-stats";
import { ActivityFeed } from "@/components/profile/activity-feed";
import { GameGrid } from "@/components/games/game-grid";
import { ReviewList } from "@/components/reviews/review-list";
import { EmptyState } from "@/components/shared/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { GameListItem } from "@/types/game";
import type { Collection } from "@/types/collection";
import type { UserPublic } from "@/types/user";

interface ProfilePageProps {
  params: { username: string };
}

export default function ProfilePage({ params }: ProfilePageProps) {
  const { data: profile, isLoading } = useQuery({
    queryKey: ["profile", params.username],
    queryFn: () =>
      apiClient.get<UserPublic>(`/users/${params.username}`),
  });

  const { data: games } = useQuery({
    queryKey: ["profile-games", params.username],
    queryFn: () =>
      apiClient.get<PaginatedResponse<GameListItem>>(
        `/users/${params.username}/games`,
        { params: { per_page: 24 } },
      ),
    enabled: !!profile,
  });

  const { data: collections } = useQuery({
    queryKey: ["profile-collections", params.username],
    queryFn: () =>
      apiClient.get<Collection[]>(
        `/users/${params.username}/collections`,
      ),
    enabled: !!profile,
  });

  if (isLoading) {
    return (
      <div className="container mx-auto max-w-5xl px-4 py-8">
        <Skeleton className="h-32 w-full rounded-lg" />
        <Skeleton className="mt-6 h-64 w-full" />
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="container mx-auto max-w-5xl px-4 py-8 text-center">
        <p className="text-muted-foreground">User not found.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <ProfileHeader user={profile} />
      <UserStats user={profile} className="mt-6" />

      <Tabs defaultValue="games" className="mt-8">
        <TabsList>
          <TabsTrigger value="games">Games</TabsTrigger>
          <TabsTrigger value="reviews">Reviews</TabsTrigger>
          <TabsTrigger value="collections">Collections</TabsTrigger>
          <TabsTrigger value="activity">Activity</TabsTrigger>
        </TabsList>

        <TabsContent value="games" className="mt-6">
          {games?.items.length === 0 ? (
            <EmptyState
              title="No games submitted"
              description={`${profile.display_name ?? profile.username} hasn't submitted any games yet.`}
            />
          ) : (
            <GameGrid games={games?.items ?? []} />
          )}
        </TabsContent>

        <TabsContent value="reviews" className="mt-6">
          <ReviewList endpoint={`/users/${params.username}/reviews`} />
        </TabsContent>

        <TabsContent value="collections" className="mt-6">
          {collections?.length === 0 ? (
            <EmptyState
              title="No public collections"
              description="This user hasn't created any public collections."
            />
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {collections?.map((collection) => (
                <a
                  key={collection.id}
                  href={`/collections/${collection.id}`}
                  className="rounded-lg border border-border p-4 transition-colors hover:border-indigo-500/50"
                >
                  <h3 className="font-semibold">{collection.name}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {collection.game_count} games
                  </p>
                </a>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="activity" className="mt-6">
          <ActivityFeed username={params.username} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
