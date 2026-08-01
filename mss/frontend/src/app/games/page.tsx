import { Suspense } from "react";

import { Skeleton } from "@/components/ui/skeleton";

import { GamesBrowseContent } from "./_components/games-browse-content";

export default function GamesPage() {
  return (
    <Suspense fallback={<GamesPageSkeleton />}>
      <GamesBrowseContent />
    </Suspense>
  );
}

function GamesPageSkeleton() {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <Skeleton className="mb-6 h-10 w-48" />
      <div className="flex gap-8">
        <Skeleton className="hidden h-96 w-[300px] lg:block" />
        <Skeleton className="h-96 flex-1" />
      </div>
    </div>
  );
}
