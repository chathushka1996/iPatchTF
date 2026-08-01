"use client";

import Link from "next/link";
import { Eye, Lock, MessageSquare, Pin } from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { formatRelativeDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { ForumThread } from "@/types/forum";

interface ThreadListProps {
  threads: ForumThread[];
  className?: string;
}

export function ThreadList({ threads, className }: ThreadListProps) {
  if (threads.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No threads yet.</p>
    );
  }

  return (
    <div className={cn("divide-y rounded-lg border", className)}>
      {threads.map((thread) => {
        const authorName =
          thread.user.display_name || thread.user.username;

        return (
          <Link
            key={thread.id}
            href={`/community/threads/${thread.slug}`}
            className="flex items-center gap-4 p-4 transition-colors hover:bg-muted/50"
          >
            <Avatar className="hidden h-9 w-9 sm:flex">
              <AvatarImage src={thread.user.avatar_url ?? undefined} />
              <AvatarFallback name={authorName} />
            </Avatar>

            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                {thread.is_pinned && (
                  <Pin className="h-3.5 w-3.5 text-primary" />
                )}
                {thread.is_locked && (
                  <Lock className="h-3.5 w-3.5 text-muted-foreground" />
                )}
                <h3 className="truncate font-medium">{thread.title}</h3>
              </div>
              <p className="mt-0.5 text-sm text-muted-foreground">
                by {authorName}
              </p>
            </div>

            <div className="hidden shrink-0 items-center gap-4 text-sm text-muted-foreground sm:flex">
              <span className="flex items-center gap-1">
                <MessageSquare className="h-3.5 w-3.5" />
                {thread.post_count}
              </span>
              <span className="flex items-center gap-1">
                <Eye className="h-3.5 w-3.5" />
                {thread.view_count}
              </span>
            </div>

            <div className="shrink-0 text-right text-xs text-muted-foreground">
              {thread.last_post_at ? (
                <span>{formatRelativeDate(thread.last_post_at)}</span>
              ) : (
                <span>{formatRelativeDate(thread.created_at)}</span>
              )}
            </div>
          </Link>
        );
      })}
    </div>
  );
}
