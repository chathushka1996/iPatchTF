export interface UserBrief {
  id: string;
  username: string;
  display_name?: string | null;
  avatar_url?: string | null;
  role?: string;
}

export interface UserPublic {
  id: string;
  username: string;
  display_name?: string | null;
  avatar_url?: string | null;
  bio?: string | null;
  website?: string | null;
  location?: string | null;
  social_discord?: string | null;
  social_twitter?: string | null;
  social_github?: string | null;
  patreon_url?: string | null;
  game_count?: number;
  review_count?: number;
  follower_count?: number;
  following_count?: number;
  created_at?: string;
}

export interface UserActivity {
  id: string;
  type: "game_submitted" | "review_posted" | "forum_post" | "follow";
  title: string;
  description?: string;
  link?: string;
  created_at: string;
}
