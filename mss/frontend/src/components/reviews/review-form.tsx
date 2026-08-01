"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { StarRating } from "@/components/reviews/star-rating";
import { cn } from "@/lib/utils";

const reviewSchema = z.object({
  score: z.number().min(1).max(10),
  body: z.string().min(1, "Review body is required"),
});

type ReviewFormData = z.infer<typeof reviewSchema>;

interface ReviewFormProps {
  onSubmit: (data: ReviewFormData) => Promise<void>;
  className?: string;
}

export function ReviewForm({ onSubmit, className }: ReviewFormProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors },
  } = useForm<ReviewFormData>({
    resolver: zodResolver(reviewSchema),
    defaultValues: { score: 0, body: "" },
  });

  const score = watch("score");

  const handleFormSubmit = async (data: ReviewFormData) => {
    setIsSubmitting(true);
    try {
      await onSubmit(data);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit(handleFormSubmit)}
      className={cn("space-y-4", className)}
    >
      <div className="space-y-2">
        <label className="text-sm font-medium">Your Rating</label>
        <StarRating
          value={score}
          onChange={(v) => setValue("score", v, { shouldValidate: true })}
        />
        {errors.score && (
          <p className="text-sm text-destructive">Please select a rating</p>
        )}
      </div>

      <Textarea
        label="Review"
        placeholder="Share your thoughts about this game..."
        autoResize
        rows={5}
        {...register("body")}
        error={errors.body?.message}
      />

      <Button type="submit" disabled={isSubmitting || score === 0}>
        {isSubmitting ? "Submitting..." : "Submit Review"}
      </Button>
    </form>
  );
}
