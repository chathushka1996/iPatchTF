"use client";

import { useState } from "react";
import { Star } from "lucide-react";

import { cn } from "@/lib/utils";

interface StarRatingProps {
  value: number;
  max?: number;
  onChange?: (value: number) => void;
  readOnly?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const SIZE_CLASSES = {
  sm: "h-4 w-4",
  md: "h-5 w-5",
  lg: "h-6 w-6",
};

export function StarRating({
  value,
  max = 10,
  onChange,
  readOnly = false,
  size = "md",
  className,
}: StarRatingProps) {
  const [hoverValue, setHoverValue] = useState(0);
  const displayValue = hoverValue || value;

  return (
    <div
      className={cn("flex items-center gap-1", className)}
      onMouseLeave={() => !readOnly && setHoverValue(0)}
    >
      {Array.from({ length: max }, (_, i) => {
        const starValue = i + 1;
        const filled = starValue <= displayValue;

        return (
          <button
            key={starValue}
            type="button"
            disabled={readOnly}
            onClick={() => onChange?.(starValue)}
            onMouseEnter={() => !readOnly && setHoverValue(starValue)}
            className={cn(
              "transition-colors",
              readOnly ? "cursor-default" : "cursor-pointer hover:scale-110",
            )}
            aria-label={`Rate ${starValue} out of ${max}`}
          >
            <Star
              className={cn(
                SIZE_CLASSES[size],
                filled
                  ? "fill-warning text-warning"
                  : "fill-transparent text-muted-foreground/40",
              )}
            />
          </button>
        );
      })}
      {value > 0 && (
        <span className="ml-2 text-sm font-medium text-muted-foreground">
          {value}/{max}
        </span>
      )}
    </div>
  );
}
