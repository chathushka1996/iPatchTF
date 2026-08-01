"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { GAME_SORT_OPTIONS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { GameSortOption } from "@/types/search";

interface SortSelectorProps {
  value: GameSortOption;
  onChange: (value: GameSortOption) => void;
  className?: string;
}

export function SortSelector({ value, onChange, className }: SortSelectorProps) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as GameSortOption)}>
      <SelectTrigger className={cn("w-[180px]", className)}>
        <SelectValue placeholder="Sort by" />
      </SelectTrigger>
      <SelectContent>
        {GAME_SORT_OPTIONS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
