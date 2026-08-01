"use client";

import { useState } from "react";
import Image from "next/image";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, X, ZoomIn } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { Screenshot } from "@/types/game";

interface GameScreenshotGalleryProps {
  screenshots: Screenshot[];
  className?: string;
}

export function GameScreenshotGallery({
  screenshots,
  className,
}: GameScreenshotGalleryProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  if (screenshots.length === 0) {
    return (
      <div
        className={cn(
          "flex aspect-video items-center justify-center rounded-lg bg-muted",
          className,
        )}
      >
        <p className="text-sm text-muted-foreground">No screenshots available</p>
      </div>
    );
  }

  const sorted = [...screenshots].sort(
    (a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0),
  );
  const active = sorted[activeIndex];

  const goTo = (index: number) => {
    setActiveIndex((index + sorted.length) % sorted.length);
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div className="group relative aspect-video overflow-hidden rounded-lg bg-muted">
        <Image
          src={active.image_url}
          alt={active.caption ?? `Screenshot ${activeIndex + 1}`}
          fill
          className="object-cover"
          sizes="(max-width: 768px) 100vw, 66vw"
          priority
        />
        <Button
          variant="secondary"
          size="icon"
          className="absolute right-3 top-3 opacity-0 transition-opacity group-hover:opacity-100"
          onClick={() => setLightboxOpen(true)}
          aria-label="Open lightbox"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        {sorted.length > 1 && (
          <>
            <Button
              variant="secondary"
              size="icon"
              className="absolute left-3 top-1/2 -translate-y-1/2"
              onClick={() => goTo(activeIndex - 1)}
              aria-label="Previous screenshot"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="secondary"
              size="icon"
              className="absolute right-3 top-1/2 -translate-y-1/2"
              onClick={() => goTo(activeIndex + 1)}
              aria-label="Next screenshot"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </>
        )}
        {active.caption && (
          <p className="absolute bottom-0 left-0 right-0 bg-black/60 px-4 py-2 text-sm text-white">
            {active.caption}
          </p>
        )}
      </div>

      {sorted.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
          {sorted.map((screenshot, index) => (
            <button
              key={screenshot.id}
              type="button"
              onClick={() => setActiveIndex(index)}
              className={cn(
                "relative h-16 w-24 shrink-0 overflow-hidden rounded-md border-2 transition-colors",
                index === activeIndex
                  ? "border-primary"
                  : "border-transparent opacity-70 hover:opacity-100",
              )}
            >
              <Image
                src={screenshot.thumbnail_url ?? screenshot.image_url}
                alt={screenshot.caption ?? `Thumbnail ${index + 1}`}
                fill
                className="object-cover"
                sizes="96px"
              />
            </button>
          ))}
        </div>
      )}

      <Dialog open={lightboxOpen} onOpenChange={setLightboxOpen}>
        <DialogContent className="max-w-5xl border-none bg-black/95 p-0">
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-2 top-2 z-10 text-white hover:bg-white/20"
            onClick={() => setLightboxOpen(false)}
          >
            <X className="h-5 w-5" />
          </Button>
          <AnimatePresence mode="wait">
            <motion.div
              key={active.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="relative aspect-video w-full"
            >
              <Image
                src={active.image_url}
                alt={active.caption ?? `Screenshot ${activeIndex + 1}`}
                fill
                className="object-contain"
                sizes="100vw"
              />
            </motion.div>
          </AnimatePresence>
        </DialogContent>
      </Dialog>
    </div>
  );
}
