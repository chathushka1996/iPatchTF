"use client";

import { Download } from "lucide-react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { MarkdownRenderer } from "@/components/shared/markdown-renderer";
import { formatDate } from "@/lib/utils";
import type { GameVersion } from "@/types/game";

interface GameVersionHistoryProps {
  versions: GameVersion[];
  className?: string;
}

export function GameVersionHistory({
  versions,
  className,
}: GameVersionHistoryProps) {
  if (versions.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No version history available.</p>
    );
  }

  return (
    <Accordion
      type="single"
      collapsible
      defaultValue={versions.find((v) => v.is_latest)?.id ?? versions[0]?.id}
      className={className}
    >
      {versions.map((version) => (
        <AccordionItem key={version.id} value={version.id}>
          <AccordionTrigger className="hover:no-underline">
            <div className="flex items-center gap-2 text-left">
              <span className="font-medium">{version.version_string}</span>
              {version.is_latest && (
                <Badge variant="success" className="text-xs">
                  Latest
                </Badge>
              )}
              {version.release_date && (
                <span className="text-sm text-muted-foreground">
                  {formatDate(version.release_date)}
                </span>
              )}
            </div>
          </AccordionTrigger>
          <AccordionContent className="space-y-4">
            {version.changelog && (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <MarkdownRenderer content={version.changelog} />
              </div>
            )}

            {version.downloads.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm font-medium">Download Mirrors</p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {version.downloads.map((dl) => (
                    <a
                      key={dl.id}
                      href={dl.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors hover:bg-muted"
                    >
                      <Download className="h-4 w-4 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">{dl.label}</p>
                        <p className="text-xs text-muted-foreground capitalize">
                          {dl.platform}
                          {dl.file_size_bytes &&
                            ` · ${formatFileSize(dl.file_size_bytes)}`}
                        </p>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
