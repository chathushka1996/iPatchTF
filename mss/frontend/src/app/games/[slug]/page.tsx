import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { GameDetailHeader } from "@/components/games/game-detail-header";
import { GameMetadataSidebar } from "@/components/games/game-metadata-sidebar";
import { GameScreenshotGallery } from "@/components/games/game-screenshot-gallery";
import { SimilarGames } from "@/components/games/similar-games";
import { ReviewList } from "@/components/reviews/review-list";
import { ReportButton } from "@/components/shared/report-button";
import { MarkdownRenderer } from "@/components/shared/markdown-renderer";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiClient } from "@/lib/api-client";
import type { Game } from "@/types/game";

import { GameActions } from "./_components/game-actions";

interface GameDetailPageProps {
  params: Promise<{ slug: string }>;
}

async function getGame(slug: string): Promise<Game | null> {
  try {
    return await apiClient.get<Game>(`/games/${slug}`, {
      next: { revalidate: 60 },
    });
  } catch {
    return null;
  }
}

export async function generateMetadata({
  params,
}: GameDetailPageProps): Promise<Metadata> {
  const { slug } = await params;
  const game = await getGame(slug);

  if (!game) {
    return { title: "Game Not Found" };
  }

  const title = `${game.title} by ${game.author.display_name ?? game.author.username}`;
  const description =
    game.synopsis?.slice(0, 160) ??
    `View details, reviews, and downloads for ${game.title} on GameVault.`;

  return {
    title,
    description,
    openGraph: {
      title: `${title} — GameVault`,
      description,
      images: game.screenshots[0]?.image_url
        ? [{ url: game.screenshots[0].image_url }]
        : undefined,
      type: "website",
    },
    twitter: {
      card: "summary_large_image",
      title: `${title} — GameVault`,
      description,
      images: game.screenshots[0]?.image_url
        ? [game.screenshots[0].image_url]
        : undefined,
    },
  };
}

export default async function GameDetailPage({ params }: GameDetailPageProps) {
  const { slug } = await params;
  const game = await getGame(slug);

  if (!game) {
    notFound();
  }

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "VideoGame",
    name: game.title,
    author: {
      "@type": "Person",
      name: game.author.display_name ?? game.author.username,
    },
    description: game.synopsis,
    aggregateRating:
      game.review_count > 0
        ? {
            "@type": "AggregateRating",
            ratingValue: game.average_score,
            reviewCount: game.review_count,
          }
        : undefined,
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <div className="container mx-auto max-w-7xl px-4 py-8">
        <div className="flex flex-col gap-8 lg:flex-row">
          {/* Main content — 70% */}
          <div className="min-w-0 flex-[7]">
            <GameDetailHeader game={game} />

            {game.screenshots.length > 0 && (
              <div className="mt-6">
                <GameScreenshotGallery screenshots={game.screenshots} />
              </div>
            )}

            <Tabs defaultValue="synopsis" className="mt-8">
              <TabsList className="w-full justify-start overflow-x-auto">
                <TabsTrigger value="synopsis">Synopsis</TabsTrigger>
                <TabsTrigger value="plot">Plot</TabsTrigger>
                <TabsTrigger value="characters">Characters</TabsTrigger>
                <TabsTrigger value="walkthrough">Walkthrough</TabsTrigger>
                <TabsTrigger value="changelog">Changelog</TabsTrigger>
              </TabsList>
              <TabsContent value="synopsis" className="mt-4">
                {game.synopsis ? (
                  <MarkdownRenderer content={game.synopsis} />
                ) : (
                  <p className="text-muted-foreground">No synopsis provided.</p>
                )}
              </TabsContent>
              <TabsContent value="plot" className="mt-4">
                {game.plot ? (
                  <MarkdownRenderer content={game.plot} />
                ) : (
                  <p className="text-muted-foreground">No plot provided.</p>
                )}
              </TabsContent>
              <TabsContent value="characters" className="mt-4">
                {game.characters ? (
                  <MarkdownRenderer content={game.characters} />
                ) : (
                  <p className="text-muted-foreground">
                    No character information provided.
                  </p>
                )}
              </TabsContent>
              <TabsContent value="walkthrough" className="mt-4">
                {game.walkthrough ? (
                  <MarkdownRenderer content={game.walkthrough} />
                ) : (
                  <p className="text-muted-foreground">
                    No walkthrough provided.
                  </p>
                )}
              </TabsContent>
              <TabsContent value="changelog" className="mt-4">
                {game.latest_version?.changelog ? (
                  <MarkdownRenderer content={game.latest_version.changelog} />
                ) : (
                  <p className="text-muted-foreground">
                    No changelog available.
                  </p>
                )}
              </TabsContent>
            </Tabs>

            <div className="mt-6 flex flex-wrap items-center gap-3">
              <GameActions game={game} />
              <ReportButton targetType="game" targetId={game.id} />
            </div>

            <section className="mt-12">
              <h2 className="mb-4 text-xl font-semibold">
                Users who liked this also liked...
              </h2>
              <SimilarGames gameId={game.id} />
            </section>

            <section className="mt-12">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-xl font-semibold">Reviews</h2>
                <Button variant="outline" size="sm">
                  Write a Review
                </Button>
              </div>
              <ReviewList
                endpoint={`/games/${game.slug}/reviews`}
                gameId={game.id}
              />
            </section>

            <div className="mt-8">
              <Button asChild variant="secondary">
                <Link href={`/games/${game.slug}/discussion`}>
                  Join the Discussion
                </Link>
              </Button>
            </div>
          </div>

          {/* Metadata sidebar — 30% */}
          <aside className="flex-[3]">
            <GameMetadataSidebar game={game} />
          </aside>
        </div>
      </div>
    </>
  );
}
