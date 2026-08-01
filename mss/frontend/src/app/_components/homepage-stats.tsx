"use client";

import { useQuery } from "@tanstack/react-query";

import { apiClient } from "@/lib/api-client";
import type { PlatformStats } from "@/types/api";

import { StatCounter } from "./stat-counter";

export function HomepageStats() {
  const { data } = useQuery({
    queryKey: ["platform-stats"],
    queryFn: () => apiClient.get<PlatformStats>("/stats"),
  });

  const stats = [
    { label: "Total Games", value: data?.total_games ?? 0 },
    { label: "Reviews", value: data?.total_reviews ?? 0 },
    { label: "Engines", value: data?.total_engines ?? 0 },
    { label: "Online Plays", value: data?.online_plays ?? 0 },
  ];

  return (
    <div className="mt-12 grid grid-cols-2 gap-4 md:grid-cols-4">
      {stats.map((stat) => (
        <StatCounter key={stat.label} label={stat.label} value={stat.value} />
      ))}
    </div>
  );
}
