import {
  Gamepad2,
  Heart,
  MessageSquare,
  Star,
} from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils";
import { cn } from "@/lib/utils";

interface UserStatsProps {
  gamesSubmitted?: number;
  reviewsWritten?: number;
  forumPosts?: number;
  likesReceived?: number;
  className?: string;
}

const STATS = [
  {
    key: "gamesSubmitted" as const,
    label: "Games Submitted",
    icon: Gamepad2,
  },
  {
    key: "reviewsWritten" as const,
    label: "Reviews Written",
    icon: Star,
  },
  {
    key: "forumPosts" as const,
    label: "Forum Posts",
    icon: MessageSquare,
  },
  {
    key: "likesReceived" as const,
    label: "Likes Received",
    icon: Heart,
  },
];

export function UserStats({
  gamesSubmitted = 0,
  reviewsWritten = 0,
  forumPosts = 0,
  likesReceived = 0,
  className,
}: UserStatsProps) {
  const values = { gamesSubmitted, reviewsWritten, forumPosts, likesReceived };

  return (
    <div
      className={cn(
        "grid grid-cols-2 gap-4 lg:grid-cols-4",
        className,
      )}
    >
      {STATS.map(({ key, label, icon: Icon }) => (
        <Card key={key}>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <Icon className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {formatNumber(values[key])}
              </p>
              <p className="text-xs text-muted-foreground">{label}</p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
