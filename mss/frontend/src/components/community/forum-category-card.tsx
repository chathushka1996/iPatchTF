"use client";

import Link from "next/link";
import { Lock, MessageSquare } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { formatRelativeDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { ForumCategory } from "@/types/forum";

interface ForumCategoryCardProps {
  category: ForumCategory;
  className?: string;
}

export function ForumCategoryCard({
  category,
  className,
}: ForumCategoryCardProps) {
  return (
    <Link href={`/community/${category.slug}`}>
      <Card
        className={cn(
          "transition-shadow hover:shadow-md",
          category.is_locked && "opacity-75",
          className,
        )}
      >
        <CardContent className="p-4">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold">{category.name}</h3>
                {category.is_locked && (
                  <Lock className="h-3.5 w-3.5 text-muted-foreground" />
                )}
              </div>
              {category.description && (
                <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                  {category.description}
                </p>
              )}
            </div>
            <MessageSquare className="h-5 w-5 shrink-0 text-muted-foreground" />
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="secondary">{category.thread_count} threads</Badge>
            <Badge variant="outline">{category.post_count} posts</Badge>
          </div>

          {category.last_post_at && (
            <p className="mt-2 text-xs text-muted-foreground">
              Last post {formatRelativeDate(category.last_post_at)}
              {category.last_post_author && (
                <>
                  {" "}
                  by{" "}
                  <span className="font-medium text-foreground">
                    {category.last_post_author.display_name ||
                      category.last_post_author.username}
                  </span>
                </>
              )}
            </p>
          )}
          {category.last_post_preview && (
            <p className="mt-1 line-clamp-1 text-xs text-muted-foreground">
              {category.last_post_preview}
            </p>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
