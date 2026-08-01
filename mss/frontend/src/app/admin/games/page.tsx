"use client";

import { useState } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { Pagination } from "@/components/shared/pagination";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { AdminGame } from "@/types/admin";
import { format } from "date-fns";
import { Search, Star, Trash2 } from "lucide-react";

export default function AdminGamesPage() {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["admin-games", search, page],
    queryFn: () =>
      apiClient.get<PaginatedResponse<AdminGame>>("/admin/games", {
        params: { q: search || undefined, page, per_page: 25 },
      }),
  });

  const approveGame = useMutation({
    mutationFn: (id: string) => apiClient.post(`/admin/games/${id}/approve`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-games"] });
    },
  });

  const featureGame = useMutation({
    mutationFn: ({ id, featured }: { id: string; featured: boolean }) =>
      apiClient.post(`/admin/games/${id}/${featured ? "feature" : "unfeature"}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-games"] });
    },
  });

  const deleteGame = useMutation({
    mutationFn: (id: string) => apiClient.delete(`/admin/games/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-games"] });
    },
  });

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Game Management</h1>
        <p className="mt-1 text-muted-foreground">
          Approve, feature, and moderate game submissions
        </p>
      </div>

      <div className="mb-6">
        <div className="relative max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search games..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="pl-9"
          />
        </div>
      </div>

      <div className="rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Title</TableHead>
              <TableHead>Author</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Approval</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i}>
                  <TableCell colSpan={6}>
                    <Skeleton className="h-10 w-full" />
                  </TableCell>
                </TableRow>
              ))
            ) : (
              data?.items.map((game) => (
                <TableRow key={game.id}>
                  <TableCell>
                    <Link
                      href={`/games/${game.slug}`}
                      className="font-medium hover:text-indigo-500"
                    >
                      {game.title}
                    </Link>
                    {game.is_featured && (
                      <Star className="ml-1 inline h-3 w-3 fill-amber-400 text-amber-400" />
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {game.author_name}
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{game.development_status}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        game.approval_status === "approved"
                          ? "default"
                          : game.approval_status === "pending"
                            ? "outline"
                            : "destructive"
                      }
                    >
                      {game.approval_status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {format(new Date(game.created_at), "MMM d, yyyy")}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-1">
                      {game.approval_status === "pending" && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => approveGame.mutate(game.id)}
                        >
                          Approve
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          featureGame.mutate({
                            id: game.id,
                            featured: !game.is_featured,
                          })
                        }
                      >
                        <Star className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteGame.mutate(game.id)}
                      >
                        <Trash2 className="h-4 w-4 text-red-500" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {data && data.pages > 1 && (
        <div className="mt-6">
          <Pagination
            page={data.page}
            totalPages={data.pages}
            basePath="/admin/games"
            onPageChange={setPage}
          />
        </div>
      )}
    </div>
  );
}
