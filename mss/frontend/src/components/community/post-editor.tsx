"use client";

import { useState } from "react";
import {
  Bold,
  Code,
  Eye,
  EyeOff,
  Image,
  Italic,
  Link,
  Quote,
  EyeOff,
} from "lucide-react";

import { MarkdownRenderer } from "@/components/shared/markdown-renderer";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

interface PostEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  onSubmit?: () => void;
  isSubmitting?: boolean;
  className?: string;
}

type ToolbarAction =
  | "bold"
  | "italic"
  | "link"
  | "image"
  | "code"
  | "quote"
  | "spoiler";

const TOOLBAR_ITEMS: {
  action: ToolbarAction;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  wrap: [string, string];
}[] = [
  { action: "bold", icon: Bold, label: "Bold", wrap: ["**", "**"] },
  { action: "italic", icon: Italic, label: "Italic", wrap: ["_", "_"] },
  { action: "link", icon: Link, label: "Link", wrap: ["[", "](url)"] },
  { action: "image", icon: Image, label: "Image", wrap: ["![alt](", ")"] },
  { action: "code", icon: Code, label: "Code", wrap: ["`", "`"] },
  { action: "quote", icon: Quote, label: "Quote", wrap: ["> ", ""] },
  { action: "spoiler", icon: EyeOff, label: "Spoiler", wrap: ["||", "||"] },
];

export function PostEditor({
  value,
  onChange,
  placeholder = "Write your post in Markdown...",
  onSubmit,
  isSubmitting = false,
  className,
}: PostEditorProps) {
  const [showPreview, setShowPreview] = useState(false);

  const applyFormat = (wrap: [string, string]) => {
    const textarea = document.activeElement as HTMLTextAreaElement | null;
    if (!textarea || textarea.tagName !== "TEXTAREA") return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selected = value.slice(start, end);
    const newValue =
      value.slice(0, start) + wrap[0] + selected + wrap[1] + value.slice(end);
    onChange(newValue);
  };

  return (
    <div className={cn("space-y-2 rounded-lg border", className)}>
      <div className="flex flex-wrap items-center gap-1 border-b p-2">
        {TOOLBAR_ITEMS.map(({ action, icon: Icon, label, wrap }) => (
          <Button
            key={action}
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => applyFormat(wrap)}
            aria-label={label}
          >
            <Icon className="h-4 w-4" />
          </Button>
        ))}
        <div className="ml-auto">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowPreview(!showPreview)}
          >
            {showPreview ? (
              <>
                <EyeOff className="mr-1 h-4 w-4" />
                Edit
              </>
            ) : (
              <>
                <Eye className="mr-1 h-4 w-4" />
                Preview
              </>
            )}
          </Button>
        </div>
      </div>

      {showPreview ? (
        <div className="prose prose-sm dark:prose-invert min-h-[200px] max-w-none p-4">
          {value ? (
            <MarkdownRenderer content={value} />
          ) : (
            <p className="text-muted-foreground">Nothing to preview</p>
          )}
        </div>
      ) : (
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          autoResize
          rows={8}
          className="border-0 focus-visible:ring-0"
        />
      )}

      {onSubmit && (
        <div className="flex justify-end border-t p-2">
          <Button onClick={onSubmit} disabled={isSubmitting || !value.trim()}>
            {isSubmitting ? "Posting..." : "Post"}
          </Button>
        </div>
      )}
    </div>
  );
}
