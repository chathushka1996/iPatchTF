"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiClient } from "@/lib/api-client";
import { useAuth } from "@/hooks/use-auth";
import type { Game } from "@/types/game";
import { Button } from "@/components/ui/button";
import { Share2 } from "lucide-react";

interface GameActionsProps {
  game: Game;
}

export function GameActions({ game }: GameActionsProps) {
  const { isAuthenticated } = useAuth();
  const queryClient = useQueryClient();

  const likeMutation = useMutation({
    mutationFn: () =>
      apiClient.post(`/games/${game.slug}/like`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["game", game.slug] });
    },
  });

  const followMutation = useMutation({
    mutationFn: () =>
      apiClient.post(`/games/${game.slug}/follow`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["game", game.slug] });
    },
  });

  const handleShare = async () => {
    const url = window.location.href;
    if (navigator.share) {
      await navigator.share({ title: game.title, url });
    } else {
      await navigator.clipboard.writeText(url);
    }
  };

  return (
    <>
      <Button
        variant="outline"
        disabled={!isAuthenticated || likeMutation.isPending}
        onClick={() => likeMutation.mutate()}
      >
        Like ({game.like_count})
      </Button>
      <Button
        variant="outline"
        disabled={!isAuthenticated || followMutation.isPending}
        onClick={() => followMutation.mutate()}
      >
        Follow
      </Button>
      <Button variant="outline" onClick={handleShare}>
        <Share2 className="mr-2 h-4 w-4" />
        Share
      </Button>
    </>
  );
}
