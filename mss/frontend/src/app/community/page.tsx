"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { ForumCategoryCard } from "@/components/community/forum-category-card";
import { EmptyState } from "@/components/shared/empty-state";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { apiClient } from "@/lib/api-client";
import type { ForumCategory } from "@/types/forum";
import { Plus } from "lucide-react";

export default function CommunityPage() {
  const { data: categories, isLoading } = useQuery({
    queryKey: ["forum-categories"],
    queryFn: () => apiClient.get<ForumCategory[]>("/forums"),
  });

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Community Forums</h1>
          <p className="mt-1 text-muted-foreground">
            Discuss games, share tips, and connect with the community
          </p>
        </div>
        <Button asChild>
          <Link href="/community/general/new">
            <Plus className="mr-2 h-4 w-4" />
            New Thread
          </Link>
        </Button>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 w-full rounded-lg" />
          ))}
        </div>
      ) : categories?.length === 0 ? (
        <EmptyState
          title="No forums yet"
          description="Forum categories will appear here once they are created."
        />
      ) : (
        <div className="space-y-3">
          {categories?.map((category) => (
            <ForumCategoryCard key={category.id} category={category} />
          ))}
        </div>
      )}
    </div>
  );
}
