"use client";

import { X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { GameFilters } from "@/types/search";

interface ActiveFilter {
  key: keyof GameFilters;
  value: string;
  label: string;
}

interface ActiveFiltersProps {
  filters: GameFilters;
  onRemove: (key: keyof GameFilters, value?: string) => void;
  onClearAll?: () => void;
  className?: string;
}

function buildActiveFilters(filters: GameFilters): ActiveFilter[] {
  const active: ActiveFilter[] = [];

  const arrayKeys: (keyof GameFilters)[] = [
    "engine",
    "status",
    "genre",
    "adult_theme",
    "transformation",
    "multimedia",
    "content_warning",
    "rating",
    "pc_gender",
  ];

  for (const key of arrayKeys) {
    const values = filters[key] as string[] | undefined;
    values?.forEach((value) => {
      active.push({ key, value, label: value.replace(/_/g, " ") });
    });
  }

  if (filters.author) {
    active.push({ key: "author", value: filters.author, label: `Author: ${filters.author}` });
  }

  if (filters.has_play_online) {
    active.push({ key: "has_play_online", value: "true", label: "Play Online" });
  }

  if (filters.min_likes) {
    active.push({
      key: "min_likes",
      value: String(filters.min_likes),
      label: `${filters.min_likes}+ likes`,
    });
  }

  if (filters.q) {
    active.push({ key: "q", value: filters.q, label: `"${filters.q}"` });
  }

  return active;
}

export function ActiveFilters({
  filters,
  onRemove,
  onClearAll,
  className,
}: ActiveFiltersProps) {
  const activeFilters = buildActiveFilters(filters);

  if (activeFilters.length === 0) return null;

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      {activeFilters.map((filter) => (
        <Badge
          key={`${filter.key}-${filter.value}`}
          variant="secondary"
          className="gap-1 pr-1"
        >
          <span className="capitalize">{filter.label}</span>
          <button
            type="button"
            onClick={() => onRemove(filter.key, filter.value)}
            className="ml-1 rounded-full p-0.5 hover:bg-muted"
            aria-label={`Remove ${filter.label} filter`}
          >
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
      {onClearAll && activeFilters.length > 1 && (
        <Button variant="ghost" size="sm" onClick={onClearAll}>
          Clear all
        </Button>
      )}
    </div>
  );
}
