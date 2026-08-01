"use client";

import Link from "next/link";
import {
  Github,
  Globe,
  MapPin,
  Twitter,
  UserPlus,
  UserMinus,
} from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { cn, formatNumber } from "@/lib/utils";
import type { UserPublic } from "@/types/user";

interface ProfileHeaderProps {
  user: UserPublic;
  isFollowing?: boolean;
  isOwnProfile?: boolean;
  onFollow?: () => void;
  onUnfollow?: () => void;
  className?: string;
}

export function ProfileHeader({
  user,
  isFollowing = false,
  isOwnProfile = false,
  onFollow,
  onUnfollow,
  className,
}: ProfileHeaderProps) {
  const displayName = user.display_name || user.username;

  return (
    <div className={cn("space-y-6", className)}>
      <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-start">
        <Avatar className="h-24 w-24">
          <AvatarImage src={user.avatar_url ?? undefined} />
          <AvatarFallback name={displayName} className="text-2xl" />
        </Avatar>

        <div className="flex-1 text-center sm:text-left">
          <h1 className="text-2xl font-bold">{displayName}</h1>
          <p className="text-muted-foreground">@{user.username}</p>
          {user.bio && (
            <p className="mt-2 text-sm leading-relaxed">{user.bio}</p>
          )}

          <div className="mt-3 flex flex-wrap items-center justify-center gap-4 text-sm sm:justify-start">
            {user.game_count !== undefined && (
              <Stat label="Games" value={user.game_count} />
            )}
            {user.review_count !== undefined && (
              <Stat label="Reviews" value={user.review_count} />
            )}
            {user.follower_count !== undefined && (
              <Stat label="Followers" value={user.follower_count} />
            )}
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-center gap-3 text-sm text-muted-foreground sm:justify-start">
            {user.location && (
              <span className="flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5" />
                {user.location}
              </span>
            )}
            {user.website && (
              <a
                href={user.website}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-primary"
              >
                <Globe className="h-3.5 w-3.5" />
                Website
              </a>
            )}
            {user.social_github && (
              <a
                href={`https://github.com/${user.social_github}`}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary"
              >
                <Github className="h-4 w-4" />
              </a>
            )}
            {user.social_twitter && (
              <a
                href={`https://twitter.com/${user.social_twitter}`}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary"
              >
                <Twitter className="h-4 w-4" />
              </a>
            )}
          </div>
        </div>

        {!isOwnProfile && (
          <div className="shrink-0">
            {isFollowing ? (
              <Button variant="outline" onClick={onUnfollow}>
                <UserMinus className="mr-2 h-4 w-4" />
                Unfollow
              </Button>
            ) : (
              <Button onClick={onFollow}>
                <UserPlus className="mr-2 h-4 w-4" />
                Follow
              </Button>
            )}
          </div>
        )}

        {isOwnProfile && (
          <Button variant="outline" asChild>
            <Link href="/settings">Edit Profile</Link>
          </Button>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <span>
      <span className="font-semibold text-foreground">
        {formatNumber(value)}
      </span>{" "}
      <span className="text-muted-foreground">{label}</span>
    </span>
  );
}
