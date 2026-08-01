"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { ThreadPost } from "@/components/community/thread-post";
import { PostEditor } from "@/components/community/post-editor";
import { Pagination } from "@/components/shared/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { ThreadDetail } from "@/types/forum";

interface ThreadPageProps {
  params: { forumSlug: string; threadSlug: string };
}

export default function ThreadPage({ params }: ThreadPageProps) {
  const searchParams = useSearchParams();
  const page = Number(searchParams.get("page") ?? 1);

  const { data: thread, isLoading } = useQuery({
    queryKey: ["thread", params.forumSlug, params.threadSlug, page],
    queryFn: () =>
      apiClient.get<ThreadDetail>(
        `/forums/${params.forumSlug}/threads/${params.threadSlug}`,
        { params: { page, per_page: 20 } },
      ),
  });

  if (isLoading) {
    return (
      <div className="container mx-auto max-w-4xl px-4 py-8">
        <Skeleton className="mb-4 h-8 w-96" />
        <div className="space-y-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (!thread) {
    return (
      <div className="container mx-auto max-w-4xl px-4 py-8 text-center">
        <p className="text-muted-foreground">Thread not found.</p>
        <Link
          href={`/community/${params.forumSlug}`}
          className="mt-4 inline-block text-indigo-500 hover:underline"
        >
          Back to forum
        </Link>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8">
      <nav className="mb-4 text-sm text-muted-foreground">
        <Link href="/community" className="hover:text-foreground">
          Community
        </Link>
        <span className="mx-2">/</span>
        <Link
          href={`/community/${params.forumSlug}`}
          className="hover:text-foreground"
        >
          Forum
        </Link>
        <span className="mx-2">/</span>
        <span className="truncate">{thread.title}</span>
      </nav>

      <h1 className="text-2xl font-bold">{thread.title}</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {thread.post_count} replies · {thread.view_count} views
      </p>

      <div className="mt-8 space-y-6">
        {thread.posts.map((post, index) => (
          <ThreadPost
            key={post.id}
            post={post}
            isOriginalPost={index === 0}
            nestedReplies
          />
        ))}
      </div>

      {!thread.is_locked && (
        <div className="mt-8 border-t border-border pt-8">
          <h2 className="mb-4 text-lg font-semibold">Reply</h2>
          <PostEditor threadId={thread.id} />
        </div>
      )}

      {thread.post_count > 20 && (
        <div className="mt-8">
          <Pagination
            page={page}
            totalPages={Math.ceil(thread.post_count / 20)}
            basePath={`/community/${params.forumSlug}/${params.threadSlug}`}
          />
        </div>
      )}
    </div>
  );
}
