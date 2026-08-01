"use client";

import Link from "next/link";
import {
  Calendar,
  Download,
  ExternalLink,
  Globe,
  Tag,
  User,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { formatDate } from "@/lib/utils";
import type { Game } from "@/types/game";

interface GameMetadataSidebarProps {
  game: Game;
  className?: string;
}

export function GameMetadataSidebar({ game, className }: GameMetadataSidebarProps) {
  const avgScore =
    typeof game.average_score === "string"
      ? parseFloat(game.average_score)
      : game.average_score;

  const tagsByCategory = game.tags.reduce<Record<string, typeof game.tags>>(
    (acc, tag) => {
      const cat = tag.category;
      if (!acc[cat]) acc[cat] = [];
      acc[cat].push(tag);
      return acc;
    },
    {},
  );

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg">Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <MetadataRow label="Engine" value={game.engine.name} />
        <MetadataRow
          label="Cost"
          value={
            game.is_free
              ? game.has_purchasable_content
                ? "Free (with IAP)"
                : "Free"
              : "Paid"
          }
        />
        <MetadataRow
          label="Rating"
          value={`${game.rating}${avgScore > 0 ? ` · ${avgScore.toFixed(1)}/10` : ""}`}
        />
        <MetadataRow label="Language" value={game.language} icon={Globe} />
        <MetadataRow
          label="PC Gender"
          value={game.original_pc_gender}
          icon={User}
          className="capitalize"
        />
        <MetadataRow
          label="Submitted"
          value={formatDate(game.created_at)}
          icon={Calendar}
        />
        <MetadataRow
          label="Updated"
          value={formatDate(game.updated_at)}
          icon={Calendar}
        />

        {game.latest_version && (
          <>
            <Separator />
            <MetadataRow
              label="Latest Version"
              value={game.latest_version.version_string}
            />
            {game.latest_version.downloads.length > 0 && (
              <div className="space-y-2">
                <p className="font-medium text-muted-foreground">Downloads</p>
                {game.latest_version.downloads.map((dl) => (
                  <a
                    key={dl.id}
                    href={dl.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 rounded-md border px-3 py-2 transition-colors hover:bg-muted"
                  >
                    <Download className="h-4 w-4 shrink-0" />
                    <span className="truncate">{dl.label}</span>
                  </a>
                ))}
              </div>
            )}
          </>
        )}

        {Object.keys(tagsByCategory).length > 0 && (
          <>
            <Separator />
            <div className="space-y-3">
              <p className="flex items-center gap-2 font-medium text-muted-foreground">
                <Tag className="h-4 w-4" />
                Tags
              </p>
              {Object.entries(tagsByCategory).map(([category, tags]) => (
                <div key={category}>
                  <p className="mb-1.5 text-xs capitalize text-muted-foreground">
                    {category.replace(/_/g, " ")}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {tags.map((tag) => (
                      <Badge key={tag.id} variant="outline" className="text-xs">
                        {tag.name}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {game.play_online_url && (
          <>
            <Separator />
            <Button className="w-full" asChild>
              <a
                href={game.play_online_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="mr-2 h-4 w-4" />
                Play Online
              </a>
            </Button>
          </>
        )}

        {game.support_url && (
          <Button variant="outline" className="w-full" asChild>
            <Link href={game.support_url} target="_blank">
              Support Page
            </Link>
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

function MetadataRow({
  label,
  value,
  icon: Icon,
  className,
}: {
  label: string;
  value: string;
  icon?: React.ComponentType<{ className?: string }>;
  className?: string;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="flex items-center gap-1.5 text-muted-foreground">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {label}
      </span>
      <span className={`text-right font-medium ${className ?? ""}`}>{value}</span>
    </div>
  );
}
