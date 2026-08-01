import Link from "next/link";
import { Suspense } from "react";

import { GameGrid } from "@/components/games/game-grid";
import { SearchBar } from "@/components/search/search-bar";
import { ReviewList } from "@/components/reviews/review-list";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

import { HomepageStats } from "./_components/homepage-stats";
import { FeaturedGamesRow } from "./_components/featured-games-row";
import { HomepageGameSection } from "./_components/homepage-game-section";

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border bg-gradient-to-b from-indigo-500/10 via-background to-background">
        <div className="container mx-auto max-w-7xl px-4 py-16 md:py-24">
          <div className="mx-auto max-w-3xl text-center">
            <h1 className="text-4xl font-bold tracking-tight text-foreground md:text-5xl lg:text-6xl">
              Discover, Share &amp; Discuss Games
            </h1>
            <p className="mt-4 text-lg text-muted-foreground">
              Explore a community-driven database of interactive games across
              every engine and genre.
            </p>
            <div className="mt-8">
              <SearchBar className="mx-auto max-w-xl" placeholder="Search games, authors, engines..." />
            </div>
            <Button asChild size="lg" className="mt-6">
              <Link href="/games">Browse All</Link>
            </Button>
          </div>
          <Suspense fallback={<StatsSkeleton />}>
            <HomepageStats />
          </Suspense>
        </div>
      </section>

      {/* Featured / Community Favorites */}
      <section className="container mx-auto max-w-7xl px-4 py-12">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Community Favorites</h2>
          <Link
            href="/games?sort=likes"
            className="text-sm text-indigo-500 hover:text-indigo-600"
          >
            View all
          </Link>
        </div>
        <Suspense fallback={<HorizontalScrollSkeleton />}>
          <FeaturedGamesRow />
        </Suspense>
      </section>

      {/* Trending This Week */}
      <section className="container mx-auto max-w-7xl px-4 py-12">
        <h2 className="mb-6 text-2xl font-semibold">Trending This Week</h2>
        <Suspense fallback={<GameGridSkeleton />}>
          <HomepageGameSection endpoint="/games/trending" limit={12} />
        </Suspense>
      </section>

      {/* Recent Submissions */}
      <section className="container mx-auto max-w-7xl px-4 py-12">
        <h2 className="mb-6 text-2xl font-semibold">Recent Submissions</h2>
        <Suspense fallback={<GameGridSkeleton />}>
          <HomepageGameSection endpoint="/games" sort="newest" limit={12} />
        </Suspense>
      </section>

      {/* Recent Updates */}
      <section className="container mx-auto max-w-7xl px-4 py-12">
        <h2 className="mb-6 text-2xl font-semibold">Recent Updates</h2>
        <Suspense fallback={<GameGridSkeleton />}>
          <HomepageGameSection endpoint="/games" sort="updated" limit={12} />
        </Suspense>
      </section>

      {/* Latest Reviews */}
      <section className="container mx-auto max-w-7xl px-4 py-12 pb-16">
        <h2 className="mb-6 text-2xl font-semibold">Latest Reviews</h2>
        <Suspense fallback={<ReviewsSkeleton />}>
          <LatestReviewsSection />
        </Suspense>
      </section>
    </div>
  );
}

function LatestReviewsSection() {
  return <ReviewList endpoint="/reviews/latest" limit={12} variant="excerpt" />;
}

function StatsSkeleton() {
  return (
    <div className="mt-12 grid grid-cols-2 gap-4 md:grid-cols-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} className="h-20 rounded-lg" />
      ))}
    </div>
  );
}

function HorizontalScrollSkeleton() {
  return (
    <div className="flex gap-4 overflow-hidden">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-48 w-40 shrink-0 rounded-lg" />
      ))}
    </div>
  );
}

function GameGridSkeleton() {
  return <GameGrid games={[]} isLoading />;
}

function ReviewsSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <Skeleton key={i} className="h-24 rounded-lg" />
      ))}
    </div>
  );
}
