"use client";

import Link from "next/link";
import {
  Gamepad2,
  MessageSquare,
  Star,
  UserPlus,
} from "lucide-react";

import { formatRelativeDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { UserActivity } from "@/types/user";

interface ActivityFeedProps {
  activities: UserActivity[];
  className?: string;
}

const ACTIVITY_ICONS: Record<
  UserActivity["type"],
  React.ComponentType<{ className?: string }>
> = {
  game_submitted: Gamepad2,
  review_posted: Star,
  forum_post: MessageSquare,
  follow: UserPlus,
};

const ACTIVITY_LABELS: Record<UserActivity["type"], string> = {
  game_submitted: "Submitted a game",
  review_posted: "Posted a review",
  forum_post: "Forum post",
  follow: "Started following",
};

export function ActivityFeed({ activities, className }: ActivityFeedProps) {
  if (activities.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No recent activity.</p>
    );
  }

  return (
    <div className={cn("relative space-y-0", className)}>
      <div className="absolute left-4 top-2 bottom-2 w-px bg-border" />
      {activities.map((activity) => {
        const Icon = ACTIVITY_ICONS[activity.type];
        const content = (
          <div className="relative flex gap-4 pb-6">
            <div className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border bg-background">
              <Icon className="h-4 w-4 text-primary" />
            </div>
            <div className="min-w-0 flex-1 pt-0.5">
              <p className="text-sm">
                <span className="text-muted-foreground">
                  {ACTIVITY_LABELS[activity.type]}
                </span>{" "}
                <span className="font-medium">{activity.title}</span>
              </p>
              {activity.description && (
                <p className="mt-0.5 text-sm text-muted-foreground line-clamp-2">
                  {activity.description}
                </p>
              )}
              <p className="mt-1 text-xs text-muted-foreground">
                {formatRelativeDate(activity.created_at)}
              </p>
            </div>
          </div>
        );

        return activity.link ? (
          <Link key={activity.id} href={activity.link} className="block hover:opacity-80">
            {content}
          </Link>
        ) : (
          <div key={activity.id}>{content}</div>
        );
      })}
    </div>
  );
}
