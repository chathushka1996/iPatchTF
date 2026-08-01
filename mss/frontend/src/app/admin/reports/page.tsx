"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { MarkdownRenderer } from "@/components/shared/markdown-renderer";
import { Pagination } from "@/components/shared/pagination";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { apiClient } from "@/lib/api-client";
import type { PaginatedResponse } from "@/types/api";
import type { AdminReport } from "@/types/admin";
import { format } from "date-fns";

export default function AdminReportsPage() {
  const [page, setPage] = useState(1);
  const [selectedReport, setSelectedReport] = useState<AdminReport | null>(null);
  const [resolutionNote, setResolutionNote] = useState("");
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["admin-reports", page],
    queryFn: () =>
      apiClient.get<PaginatedResponse<AdminReport>>("/admin/reports", {
        params: { page, per_page: 25, status: "pending" },
      }),
  });

  const updateReport = useMutation({
    mutationFn: ({
      id,
      status,
      note,
    }: {
      id: string;
      status: string;
      note?: string;
    }) =>
      apiClient.patch(`/admin/reports/${id}`, {
        status,
        resolution_note: note,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-reports"] });
      setSelectedReport(null);
      setResolutionNote("");
    },
  });

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Reports Queue</h1>
        <p className="mt-1 text-muted-foreground">
          Review and moderate reported content
        </p>
      </div>

      <div className="rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Reporter</TableHead>
              <TableHead>Target</TableHead>
              <TableHead>Reason</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Date</TableHead>
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
            ) : data?.items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} className="py-8 text-center text-muted-foreground">
                  No pending reports
                </TableCell>
              </TableRow>
            ) : (
              data?.items.map((report) => (
                <TableRow key={report.id}>
                  <TableCell>{report.reporter_username}</TableCell>
                  <TableCell>
                    <span className="capitalize">{report.target_type}</span>
                    <span className="text-muted-foreground">
                      {" "}
                      · {report.target_preview?.slice(0, 40)}...
                    </span>
                  </TableCell>
                  <TableCell>{report.reason}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{report.status}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {format(new Date(report.created_at), "MMM d, yyyy")}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedReport(report)}
                    >
                      Review
                    </Button>
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
            basePath="/admin/reports"
            onPageChange={setPage}
          />
        </div>
      )}

      <Dialog
        open={!!selectedReport}
        onOpenChange={() => setSelectedReport(null)}
      >
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Review Report</DialogTitle>
          </DialogHeader>
          {selectedReport && (
            <div className="space-y-4">
              <div className="rounded-lg border border-border p-4">
                <p className="text-sm font-medium">Reported Content</p>
                <div className="mt-2 text-sm">
                  {selectedReport.target_content ? (
                    <MarkdownRenderer content={selectedReport.target_content} />
                  ) : (
                    <p className="text-muted-foreground">
                      {selectedReport.target_preview}
                    </p>
                  )}
                </div>
              </div>
              <div>
                <p className="text-sm font-medium">Reason</p>
                <p className="text-sm text-muted-foreground">
                  {selectedReport.reason}
                </p>
                {selectedReport.description && (
                  <p className="mt-1 text-sm">{selectedReport.description}</p>
                )}
              </div>
              <Textarea
                placeholder="Resolution note (optional)"
                value={resolutionNote}
                onChange={(e) => setResolutionNote(e.target.value)}
              />
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() =>
                    updateReport.mutate({
                      id: selectedReport.id,
                      status: "dismissed",
                      note: resolutionNote,
                    })
                  }
                >
                  Dismiss
                </Button>
                <Button
                  variant="destructive"
                  onClick={() =>
                    updateReport.mutate({
                      id: selectedReport.id,
                      status: "resolved",
                      note: resolutionNote,
                    })
                  }
                >
                  Resolve &amp; Remove
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
