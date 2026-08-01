"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { ThreadList } from "@/components/community/thread-list";
import { Pagination } from "@/components/shared/pagination";
import { EmptyState } from "@/components/shared/empty-state";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { ForumCategory, Thread } from "@/types/forum";
import { Plus } from "lucide-react";

interface ForumCategoryPageProps {
  params: { forumSlug: string };
}

export default function ForumCategoryPage({ params }: ForumCategoryPageProps) {
  const searchParams = useSearchParams();
  const page = Number(searchParams.get("page") ?? 1);

  const { data: forum, isLoading: forumLoading } = useQuery({
    queryKey: ["forum", params.forumSlug],
    queryFn: () =>
      apiClient.get<ForumCategory>(`/forums/${params.forumSlug}`),
  });

  const { data: threads, isLoading: threadsLoading } = useQuery({
    queryKey: ["forum-threads", params.forumSlug, page],
    queryFn: () =>
      apiClient.get<PaginatedResponse<Thread>>(
        `/forums/${params.forumSlug}/threads`,
        { params: { page, per_page: 25 } },
      ),
    enabled: !!forum,
  });

  const isLoading = forumLoading || threadsLoading;

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          {isLoading ? (
            <>
              <Skeleton className="h-8 w-48" />
              <Skeleton className="mt-2 h-4 w-72" />
            </>
          ) : (
            <>
              <h1 className="text-3xl font-bold">{forum?.name}</h1>
              {forum?.description && (
                <p className="mt-1 text-muted-foreground">
                  {forum.description}
                </p>
              )}
            </>
          )}
        </div>
        <Button asChild disabled={forum?.is_locked}>
          <Link href={`/community/${params.forumSlug}/new`}>
            <Plus className="mr-2 h-4 w-4" />
            New Thread
          </Link>
        </Button>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 10 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full rounded-lg" />
          ))}
        </div>
      ) : threads?.items.length === 0 ? (
        <EmptyState
          title="No threads yet"
          description="Be the first to start a discussion in this forum."
          actionLabel="New Thread"
          actionHref={`/community/${params.forumSlug}/new`}
        />
      ) : (
        <ThreadList
          threads={threads?.items ?? []}
          forumSlug={params.forumSlug}
          variant="list"
        />
      )}

      {threads && threads.pages > 1 && (
        <div className="mt-8">
          <Pagination
            page={threads.page}
            totalPages={threads.pages}
            basePath={`/community/${params.forumSlug}`}
          />
        </div>
      )}
    </div>
  );
}
