"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { ThreadList } from "@/components/community/thread-list";
import { PostEditor } from "@/components/community/post-editor";
import { Pagination } from "@/components/shared/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { ThreadDetail } from "@/types/forum";
import type { Game } from "@/types/game";

interface GameDiscussionPageProps {
  params: { slug: string };
}

export default function GameDiscussionPage({
  params,
}: GameDiscussionPageProps) {
  const { data: game, isLoading: gameLoading } = useQuery({
    queryKey: ["game", params.slug],
    queryFn: () => apiClient.get<Game>(`/games/${params.slug}`),
  });

  const { data: thread, isLoading: threadLoading } = useQuery({
    queryKey: ["game-discussion", params.slug],
    queryFn: () =>
      apiClient.get<ThreadDetail>(`/games/${params.slug}/discussion`),
    enabled: !!game,
  });

  if (gameLoading || threadLoading) {
    return (
      <div className="container mx-auto max-w-4xl px-4 py-8">
        <Skeleton className="mb-4 h-8 w-64" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!game || !thread) {
    return (
      <div className="container mx-auto max-w-4xl px-4 py-8 text-center">
        <p className="text-muted-foreground">Discussion thread not found.</p>
        <Link
          href={`/games/${params.slug}`}
          className="mt-4 inline-block text-indigo-500 hover:underline"
        >
          Back to game
        </Link>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <nav className="mb-4 text-sm text-muted-foreground">
        <Link href={`/games/${params.slug}`} className="hover:text-foreground">
          {game.title}
        </Link>
        <span className="mx-2">/</span>
        <span>Discussion</span>
      </nav>

      <h1 className="text-2xl font-bold">{thread.title}</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {thread.post_count} posts · {thread.view_count} views
      </p>

      <div className="mt-8">
        <ThreadList
          thread={thread}
          showOriginalPost
          nestedReplies
        />
      </div>

      {!thread.is_locked && (
        <div className="mt-8 border-t border-border pt-8">
          <h2 className="mb-4 text-lg font-semibold">Reply</h2>
          <PostEditor
            threadId={thread.id}
            onSuccess={() => {
              /* refetch handled by component */
            }}
          />
        </div>
      )}

      {thread.post_count > 20 && (
        <div className="mt-8">
          <Pagination
            page={1}
            totalPages={Math.ceil(thread.post_count / 20)}
            basePath={`/games/${params.slug}/discussion`}
          />
        </div>
      )}
    </div>
  );
}
