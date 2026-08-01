export type DevelopmentStatus =
  | "concept"
  | "demo"
  | "alpha"
  | "beta"
  | "complete"
  | "discontinued";

export type ContentRating = "G" | "PG" | "R" | "X" | "XXX";

export type PCGender =
  | "male"
  | "female"
  | "selectable"
  | "genderless"
  | "hermaphrodite";

export interface Engine {
  id: number;
  name: string;
  slug: string;
  game_count?: number;
}

export interface Tag {
  id: number;
  name: string;
  slug: string;
  category: string;
}

export interface Download {
  id: string;
  url: string;
  label: string;
  file_size_bytes?: number | null;
  platform: string;
}

export interface Screenshot {
  id: string;
  image_url: string;
  thumbnail_url?: string | null;
  caption?: string | null;
  sort_order?: number;
}

export interface GameVersion {
  id: string;
  version_string: string;
  changelog?: string | null;
  release_date?: string | null;
  is_latest?: boolean;
  downloads: Download[];
  created_at: string;
}

export interface GameListItem {
  id: string;
  title: string;
  slug: string;
  engine_name: string;
  author_name: string;
  development_status: string;
  rating: string;
  like_count: number;
  review_count: number;
  average_score: number | string;
  thumbnail_url?: string | null;
  created_at: string;
  updated_at: string;
}

export interface Game extends GameListItem {
  synopsis?: string | null;
  plot?: string | null;
  characters?: string | null;
  walkthrough?: string | null;
  engine_id: number;
  original_pc_gender: string;
  is_free: boolean;
  has_purchasable_content: boolean;
  support_url?: string | null;
  language: string;
  play_online_url?: string | null;
  view_count: number;
  engine: Engine;
  author: {
    id: string;
    username: string;
    display_name?: string | null;
    avatar_url?: string | null;
  };
  tags: Tag[];
  screenshots: Screenshot[];
  latest_version?: GameVersion | null;
}
