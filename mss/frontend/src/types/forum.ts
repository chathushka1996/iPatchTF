import type { UserBrief } from "./user";

export interface ForumCategory {
  id: number;
  name: string;
  slug: string;
  description?: string | null;
  sort_order?: number;
  is_locked?: boolean;
  thread_count: number;
  post_count: number;
  last_post_at?: string | null;
  last_post_preview?: string | null;
  last_post_author?: UserBrief | null;
}

export interface ForumThread {
  id: string;
  forum_category_id: number;
  game_id?: string | null;
  user: UserBrief;
  title: string;
  slug: string;
  is_pinned: boolean;
  is_locked: boolean;
  view_count: number;
  post_count: number;
  last_post_at?: string | null;
  created_at: string;
}

export interface ForumPost {
  id: string;
  thread_id: string;
  user: UserBrief;
  parent_id?: string | null;
  body: string;
  body_html?: string | null;
  is_edited: boolean;
  edited_at?: string | null;
  created_at: string;
  replies?: ForumPost[];
}

export interface ChatMessage {
  id: string;
  user: UserBrief;
  body: string;
  created_at: string;
}
