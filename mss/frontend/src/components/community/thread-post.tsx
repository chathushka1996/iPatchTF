"use client";

import Link from "next/link";
import {
  Edit,
  Flag,
  MessageSquare,
  Quote,
  Trash2,
} from "lucide-react";

import { MarkdownRenderer } from "@/components/shared/markdown-renderer";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { formatRelativeDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { ForumPost } from "@/types/forum";

interface ThreadPostProps {
  post: ForumPost;
  isAuthor?: boolean;
  isModerator?: boolean;
  onReply?: (postId: string) => void;
  onQuote?: (postId: string) => void;
  onEdit?: (postId: string) => void;
  onDelete?: (postId: string) => void;
  onReport?: (postId: string) => void;
  className?: string;
}

const ROLE_VARIANTS: Record<string, "default" | "secondary" | "destructive"> = {
  admin: "destructive",
  moderator: "default",
  user: "secondary",
};

export function ThreadPost({
  post,
  isAuthor = false,
  isModerator = false,
  onReply,
  onQuote,
  onEdit,
  onDelete,
  onReport,
  className,
}: ThreadPostProps) {
  const displayName = post.user.display_name || post.user.username;
  const role = post.user.role ?? "user";

  return (
    <Card className={cn(className)} id={`post-${post.id}`}>
      <CardContent className="p-4">
        <div className="flex gap-3">
          <Link href={`/users/${post.user.username}`}>
            <Avatar className="h-10 w-10">
              <AvatarImage src={post.user.avatar_url ?? undefined} />
              <AvatarFallback name={displayName} />
            </Avatar>
          </Link>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Link
                href={`/users/${post.user.username}`}
                className="font-medium hover:text-primary"
              >
                {displayName}
              </Link>
              {role !== "user" && (
                <Badge variant={ROLE_VARIANTS[role] ?? "secondary"} className="text-xs capitalize">
                  {role}
                </Badge>
              )}
              <span className="text-xs text-muted-foreground">
                {formatRelativeDate(post.created_at)}
              </span>
              {post.is_edited && (
                <span className="text-xs text-muted-foreground">(edited)</span>
              )}
            </div>

            <div className="prose prose-sm dark:prose-invert mt-3 max-w-none">
              <MarkdownRenderer
                content={post.body_html ?? post.body}
              />
            </div>

            <div className="mt-3 flex flex-wrap gap-1">
              {onReply && (
                <Button variant="ghost" size="sm" onClick={() => onReply(post.id)}>
                  <MessageSquare className="mr-1 h-3.5 w-3.5" />
                  Reply
                </Button>
              )}
              {onQuote && (
                <Button variant="ghost" size="sm" onClick={() => onQuote(post.id)}>
                  <Quote className="mr-1 h-3.5 w-3.5" />
                  Quote
                </Button>
              )}
              {isAuthor && onEdit && (
                <Button variant="ghost" size="sm" onClick={() => onEdit(post.id)}>
                  <Edit className="mr-1 h-3.5 w-3.5" />
                  Edit
                </Button>
              )}
              {(isAuthor || isModerator) && onDelete && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onDelete(post.id)}
                >
                  <Trash2 className="mr-1 h-3.5 w-3.5" />
                  Delete
                </Button>
              )}
              {onReport && !isAuthor && (
                <Button variant="ghost" size="sm" onClick={() => onReport(post.id)}>
                  <Flag className="mr-1 h-3.5 w-3.5" />
                  Report
                </Button>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
