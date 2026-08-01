import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";

import { ResetPasswordForm } from "@/components/auth/reset-password-form";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export const metadata: Metadata = {
  title: "Reset Password",
  description: "Set a new password for your GameVault account",
};

export default function ResetPasswordPage() {
  return (
    <div className="container mx-auto flex min-h-[calc(100vh-8rem)] max-w-md items-center justify-center px-4 py-12">
      <Card className="w-full">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Reset password</CardTitle>
          <CardDescription>Enter your new password below</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Suspense fallback={<Skeleton className="h-48 w-full" />}>
            <ResetPasswordForm />
          </Suspense>
          <p className="text-center text-sm text-muted-foreground">
            <Link
              href="/login"
              className="text-indigo-500 hover:text-indigo-600 hover:underline"
            >
              Back to login
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
