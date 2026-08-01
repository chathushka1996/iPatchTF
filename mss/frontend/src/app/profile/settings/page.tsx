"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { AvatarUpload } from "@/components/profile/avatar-upload";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import type { NotificationPreferences, User } from "@/types/user";

export default function ProfileSettingsPage() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();
  const queryClient = useQueryClient();

  const { data: user, isLoading } = useQuery({
    queryKey: ["me"],
    queryFn: () => apiClient.get<User>("/users/me"),
    enabled: isAuthenticated,
  });

  const { data: preferences } = useQuery({
    queryKey: ["notification-preferences"],
    queryFn: () =>
      apiClient.get<NotificationPreferences>(
        "/users/me/notification-preferences",
      ),
    enabled: isAuthenticated,
  });

  const updateProfile = useMutation({
    mutationFn: (data: Partial<User>) =>
      apiClient.patch<User>("/users/me", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["me"] });
    },
  });

  const changePassword = useMutation({
    mutationFn: (data: { current_password: string; new_password: string }) =>
      apiClient.post("/users/me/change-password", data),
  });

  const updatePreferences = useMutation({
    mutationFn: (data: NotificationPreferences) =>
      apiClient.patch("/users/me/notification-preferences", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notification-preferences"] });
    },
  });

  const deleteAccount = useMutation({
    mutationFn: () => apiClient.delete("/users/me"),
    onSuccess: () => {
      router.push("/");
    },
  });

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push("/login?redirect=/profile/settings");
    }
  }, [isAuthenticated, authLoading, router]);

  if (authLoading || isLoading) {
    return (
      <div className="container mx-auto max-w-2xl px-4 py-8">
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="container mx-auto max-w-2xl space-y-8 px-4 py-8">
      <div>
        <h1 className="text-3xl font-bold">Account Settings</h1>
        <p className="mt-1 text-muted-foreground">
          Manage your profile and preferences
        </p>
      </div>

      {/* Profile */}
      <Card>
        <CardHeader>
          <CardTitle>Profile</CardTitle>
          <CardDescription>Update your public profile information</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <AvatarUpload currentUrl={user.avatar_url} />
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              const form = e.currentTarget;
              const formData = new FormData(form);
              updateProfile.mutate({
                display_name: formData.get("display_name") as string,
                bio: formData.get("bio") as string,
                website: formData.get("website") as string,
                location: formData.get("location") as string,
              });
            }}
          >
            <div className="space-y-2">
              <Label htmlFor="display_name">Display Name</Label>
              <Input
                id="display_name"
                name="display_name"
                defaultValue={user.display_name ?? ""}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bio">Bio</Label>
              <Textarea
                id="bio"
                name="bio"
                defaultValue={user.bio ?? ""}
                rows={4}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="website">Website</Label>
                <Input
                  id="website"
                  name="website"
                  type="url"
                  defaultValue={user.website ?? ""}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="location">Location</Label>
                <Input
                  id="location"
                  name="location"
                  defaultValue={user.location ?? ""}
                />
              </div>
            </div>
            <Button type="submit" disabled={updateProfile.isPending}>
              Save Changes
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Password */}
      <Card>
        <CardHeader>
          <CardTitle>Password</CardTitle>
          <CardDescription>Change your account password</CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              const form = e.currentTarget;
              const formData = new FormData(form);
              changePassword.mutate({
                current_password: formData.get("current_password") as string,
                new_password: formData.get("new_password") as string,
              });
            }}
          >
            <div className="space-y-2">
              <Label htmlFor="current_password">Current Password</Label>
              <Input
                id="current_password"
                name="current_password"
                type="password"
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="new_password">New Password</Label>
              <Input
                id="new_password"
                name="new_password"
                type="password"
                required
                minLength={8}
              />
            </div>
            <Button type="submit" disabled={changePassword.isPending}>
              Update Password
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Notifications */}
      <Card>
        <CardHeader>
          <CardTitle>Notifications</CardTitle>
          <CardDescription>
            Choose which notifications you receive via email
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {preferences &&
            Object.entries(preferences).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <Label htmlFor={key} className="capitalize">
                  {key.replace(/_/g, " ")}
                </Label>
                <Switch
                  id={key}
                  checked={value}
                  onCheckedChange={(checked) =>
                    updatePreferences.mutate({
                      ...preferences,
                      [key]: checked,
                    })
                  }
                />
              </div>
            ))}
        </CardContent>
      </Card>

      {/* 2FA */}
      <Card>
        <CardHeader>
          <CardTitle>Two-Factor Authentication</CardTitle>
          <CardDescription>
            Add an extra layer of security to your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="outline">Set up 2FA</Button>
        </CardContent>
      </Card>

      {/* Delete account */}
      <Card className="border-red-500/20">
        <CardHeader>
          <CardTitle className="text-red-500">Danger Zone</CardTitle>
          <CardDescription>
            Permanently delete your account and all associated data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ConfirmDialog
            title="Delete Account"
            description="This action cannot be undone. All your games, reviews, and data will be permanently removed."
            confirmLabel="Delete Account"
            variant="destructive"
            onConfirm={() => deleteAccount.mutate()}
          >
            <Button variant="destructive" disabled={deleteAccount.isPending}>
              Delete Account
            </Button>
          </ConfirmDialog>
        </CardContent>
      </Card>
    </div>
  );
}
