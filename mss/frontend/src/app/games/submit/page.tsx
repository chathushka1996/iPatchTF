"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { GameSubmitForm } from "@/components/games/game-submit-form";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";

export default function GameSubmitPage() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login?redirect=/games/submit");
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) {
    return (
      <div className="container mx-auto max-w-5xl px-4 py-8">
        <Skeleton className="mb-6 h-10 w-64" />
        <Skeleton className="h-[600px] w-full" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Submit a Game</h1>
        <p className="mt-1 text-muted-foreground">
          Share your game with the GameVault community
        </p>
      </div>
      <GameSubmitForm mode="create" />
    </div>
  );
}
