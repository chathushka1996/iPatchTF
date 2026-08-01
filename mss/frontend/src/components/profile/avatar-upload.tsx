"use client";

import { useRef, useState } from "react";
import { Camera, X } from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { MAX_UPLOAD_SIZES } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface AvatarUploadProps {
  currentUrl?: string | null;
  name?: string;
  onUpload: (file: File) => Promise<void>;
  className?: string;
}

export function AvatarUpload({
  currentUrl,
  name = "User",
  onUpload,
  className,
}: AvatarUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const displayUrl = preview ?? currentUrl;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select an image file");
      return;
    }

    if (file.size > MAX_UPLOAD_SIZES.avatar) {
      setError("Image must be under 2MB");
      return;
    }

    setError(null);
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);

    setIsUploading(true);
    try {
      await onUpload(file);
    } catch {
      setPreview(null);
      setError("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const clearPreview = () => {
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className={cn("flex flex-col items-center gap-4", className)}>
      <div className="relative">
        <Avatar className="h-24 w-24">
          {displayUrl ? (
            <AvatarImage src={displayUrl} alt={name} />
          ) : null}
          <AvatarFallback name={name} className="text-2xl" />
        </Avatar>
        {preview && (
          <button
            type="button"
            onClick={clearPreview}
            className="absolute -right-1 -top-1 rounded-full bg-destructive p-1 text-destructive-foreground"
            aria-label="Remove preview"
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      <Button
        variant="outline"
        size="sm"
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
      >
        {isUploading ? (
          "Uploading..."
        ) : (
          <>
            <Camera className="mr-2 h-4 w-4" />
            Change Avatar
          </>
        )}
      </Button>

      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
