"use client";

import { useCallback, useMemo, useState } from "react";

import { useDebounce } from "@/hooks/use-debounce";
import type { GameSearchFilters, GameSortOption } from "@/types/game";

const defaultFilters: GameSearchFilters = {
  q: "",
  engine: [],
  status: [],
  genre: [],
  adult_theme: [],
  transformation: [],
  multimedia: [],
  content_warning: [],
  rating: [],
  pc_gender: [],
  author: "",
  has_play_online: null,
  min_likes: null,
  sort: "newest",
};

export function useSearch(initialFilters?: Partial<GameSearchFilters>) {
  const [query, setQuery] = useState(initialFilters?.q ?? "");
  const [filters, setFilters] = useState<GameSearchFilters>({
    ...defaultFilters,
    ...initialFilters,
  });

  const debouncedQuery = useDebounce(query, 300);

  const searchParams = useMemo(
    () => ({
      q: debouncedQuery || undefined,
      engine: filters.engine.length ? filters.engine : undefined,
      status: filters.status.length ? filters.status : undefined,
      genre: filters.genre.length ? filters.genre : undefined,
      adult_theme: filters.adult_theme.length ? filters.adult_theme : undefined,
      transformation: filters.transformation.length
        ? filters.transformation
        : undefined,
      multimedia: filters.multimedia.length ? filters.multimedia : undefined,
      content_warning: filters.content_warning.length
        ? filters.content_warning
        : undefined,
      rating: filters.rating.length ? filters.rating : undefined,
      pc_gender: filters.pc_gender.length ? filters.pc_gender : undefined,
      author: filters.author || undefined,
      has_play_online: filters.has_play_online ?? undefined,
      min_likes: filters.min_likes ?? undefined,
      sort: filters.sort,
    }),
    [debouncedQuery, filters],
  );

  const updateFilter = useCallback(
    <K extends keyof GameSearchFilters>(key: K, value: GameSearchFilters[K]) => {
      setFilters((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const toggleArrayFilter = useCallback(
    <K extends keyof Pick<
      GameSearchFilters,
      | "engine"
      | "status"
      | "genre"
      | "adult_theme"
      | "transformation"
      | "multimedia"
      | "content_warning"
      | "rating"
      | "pc_gender"
    >>(key: K, value: GameSearchFilters[K][number]) => {
      setFilters((prev) => {
        const current = prev[key] as string[];
        const next = current.includes(value as string)
          ? current.filter((v) => v !== value)
          : [...current, value as string];
        return { ...prev, [key]: next };
      });
    },
    [],
  );

  const setSort = useCallback((sort: GameSortOption) => {
    setFilters((prev) => ({ ...prev, sort }));
  }, []);

  const resetFilters = useCallback(() => {
    setQuery("");
    setFilters(defaultFilters);
  }, []);

  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.engine.length) count++;
    if (filters.status.length) count++;
    if (filters.genre.length) count++;
    if (filters.adult_theme.length) count++;
    if (filters.transformation.length) count++;
    if (filters.multimedia.length) count++;
    if (filters.content_warning.length) count++;
    if (filters.rating.length) count++;
    if (filters.pc_gender.length) count++;
    if (filters.author) count++;
    if (filters.has_play_online !== null) count++;
    if (filters.min_likes !== null) count++;
    return count;
  }, [filters]);

  return {
    query,
    setQuery,
    debouncedQuery,
    filters,
    setFilters,
    searchParams,
    updateFilter,
    toggleArrayFilter,
    setSort,
    resetFilters,
    activeFilterCount,
  };
}
