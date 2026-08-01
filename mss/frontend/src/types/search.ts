import type { ContentRating, DevelopmentStatus, PCGender } from "./game";

export type GameSortOption =
  | "newest"
  | "updated"
  | "rating"
  | "likes"
  | "title"
  | "trending";

export interface GameFilters {
  q?: string;
  engine?: string[];
  status?: DevelopmentStatus[];
  genre?: string[];
  adult_theme?: string[];
  transformation?: string[];
  multimedia?: string[];
  content_warning?: string[];
  rating?: ContentRating[];
  pc_gender?: PCGender[];
  author?: string;
  has_play_online?: boolean;
  min_likes?: number;
  sort?: GameSortOption;
  page?: number;
  per_page?: number;
}

export interface SearchSuggestion {
  id: string;
  label: string;
  type: "game" | "author" | "engine";
  href: string;
}
