import { Suspense } from "react";

import { Skeleton } from "@/components/ui/skeleton";

import { BrowseCategoryContent } from "./_components/browse-category-content";

interface BrowseCategoryPageProps {
  params: { category: string };
}

export default function BrowseCategoryPage({ params }: BrowseCategoryPageProps) {
  return (
    <Suspense fallback={<BrowseSkeleton />}>
      <BrowseCategoryContent category={params.category} />
    </Suspense>
  );
}

function BrowseSkeleton() {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-8">
      <Skeleton className="mb-6 h-10 w-64" />
      <Skeleton className="h-96 w-full" />
    </div>
  );
}
