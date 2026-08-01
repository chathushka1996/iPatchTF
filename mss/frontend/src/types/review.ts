import type { UserBrief } from "./user";

export interface Review {
  id: string;
  game_id: string;
  user: UserBrief;
  version_reviewed?: string | null;
  score: number;
  body: string;
  helpful_count: number;
  not_helpful_count: number;
  is_edited: boolean;
  created_at: string;
  updated_at: string;
}

export interface ReviewCreate {
  score: number;
  body: string;
}
