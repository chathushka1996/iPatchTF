"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { GameSubmitForm } from "@/components/games/game-submit-form";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { apiClient } from "@/lib/api-client";
import type { Game } from "@/types/game";

interface EditGamePageProps {
  params: { slug: string };
}

export default function EditGamePage({ params }: EditGamePageProps) {
  const { isAuthenticated, isLoading: authLoading, user } = useAuth();
  const router = useRouter();

  const { data: game, isLoading: gameLoading } = useQuery({
    queryKey: ["game", params.slug],
    queryFn: () => apiClient.get<Game>(`/games/${params.slug}`),
    enabled: isAuthenticated,
  });

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push(`/login?redirect=/games/${params.slug}/edit`);
    }
  }, [isAuthenticated, authLoading, router, params.slug]);

  useEffect(() => {
    if (
      game &&
      user &&
      game.author.id !== user.id &&
      user.role !== "admin"
    ) {
      router.push(`/games/${params.slug}`);
    }
  }, [game, user, router, params.slug]);

  if (authLoading || gameLoading) {
    return (
      <div className="container mx-auto max-w-5xl px-4 py-8">
        <Skeleton className="mb-6 h-10 w-64" />
        <Skeleton className="h-[600px] w-full" />
      </div>
    );
  }

  if (!game) {
    return null;
  }

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Edit Game</h1>
        <p className="mt-1 text-muted-foreground">
          Update details for {game.title}
        </p>
      </div>
      <GameSubmitForm mode="edit" initialData={game} />
    </div>
  );
}
