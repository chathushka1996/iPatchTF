"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  CONTENT_RATINGS,
  DEVELOPMENT_STATUSES,
  PC_GENDERS,
} from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Engine } from "@/types/game";

const gameSchema = z.object({
  title: z.string().min(1, "Title is required").max(255),
  engine_id: z.string().min(1, "Engine is required"),
  development_status: z.string().min(1, "Status is required"),
  rating: z.string().min(1, "Rating is required"),
  original_pc_gender: z.string().min(1, "PC gender is required"),
  language: z.string().default("English"),
  is_free: z.boolean().default(true),
  has_purchasable_content: z.boolean().default(false),
  synopsis: z.string().optional(),
  plot: z.string().optional(),
  play_online_url: z.string().url().optional().or(z.literal("")),
});

type GameFormData = z.infer<typeof gameSchema>;

const STEPS = [
  { id: "basics", title: "Basic Info" },
  { id: "content", title: "Content" },
  { id: "details", title: "Details" },
  { id: "review", title: "Review" },
];

interface GameSubmitFormProps {
  engines: Engine[];
  onSubmit: (data: GameFormData) => Promise<void>;
  className?: string;
}

export function GameSubmitForm({
  engines,
  onSubmit,
  className,
}: GameSubmitFormProps) {
  const [step, setStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    trigger,
    formState: { errors },
  } = useForm<GameFormData>({
    resolver: zodResolver(gameSchema),
    defaultValues: {
      language: "English",
      is_free: true,
      has_purchasable_content: false,
    },
  });

  const formData = watch();

  const nextStep = async () => {
    const fieldsToValidate: (keyof GameFormData)[] =
      step === 0
        ? ["title", "engine_id", "development_status"]
        : step === 1
          ? ["rating", "original_pc_gender"]
          : [];

    const valid = await trigger(fieldsToValidate);
    if (valid) setStep((s) => Math.min(s + 1, STEPS.length - 1));
  };

  const prevStep = () => setStep((s) => Math.max(s - 1, 0));

  const handleFormSubmit = async (data: GameFormData) => {
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
      className={cn("space-y-6", className)}
    >
      <div className="flex items-center justify-between">
        {STEPS.map((s, i) => (
          <div
            key={s.id}
            className={cn(
              "flex items-center gap-2 text-sm",
              i <= step ? "text-primary" : "text-muted-foreground",
            )}
          >
            <span
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full border-2 text-xs font-medium",
                i <= step
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-muted",
              )}
            >
              {i + 1}
            </span>
            <span className="hidden sm:inline">{s.title}</span>
          </div>
        ))}
      </div>

      {step === 0 && (
        <div className="space-y-4">
          <Input
            label="Title"
            {...register("title")}
            error={errors.title?.message}
          />
          <div className="space-y-2">
            <Label>Engine</Label>
            <Select
              value={formData.engine_id}
              onValueChange={(v) => setValue("engine_id", v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select engine" />
              </SelectTrigger>
              <SelectContent>
                {engines.map((engine) => (
                  <SelectItem key={engine.id} value={String(engine.id)}>
                    {engine.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.engine_id && (
              <p className="text-sm text-destructive">
                {errors.engine_id.message}
              </p>
            )}
          </div>
          <div className="space-y-2">
            <Label>Development Status</Label>
            <Select
              value={formData.development_status}
              onValueChange={(v) => setValue("development_status", v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select status" />
              </SelectTrigger>
              <SelectContent>
                {DEVELOPMENT_STATUSES.map((status) => (
                  <SelectItem key={status.value} value={status.value}>
                    {status.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      )}

      {step === 1 && (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label>Content Rating</Label>
            <Select
              value={formData.rating}
              onValueChange={(v) => setValue("rating", v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select rating" />
              </SelectTrigger>
              <SelectContent>
                {CONTENT_RATINGS.map((rating) => (
                  <SelectItem key={rating.value} value={rating.value}>
                    {rating.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Original PC Gender</Label>
            <Select
              value={formData.original_pc_gender}
              onValueChange={(v) => setValue("original_pc_gender", v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select gender" />
              </SelectTrigger>
              <SelectContent>
                {PC_GENDERS.map((gender) => (
                  <SelectItem key={gender.value} value={gender.value}>
                    {gender.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Textarea
            label="Synopsis"
            {...register("synopsis")}
            autoResize
            rows={4}
          />
          <Textarea
            label="Plot"
            {...register("plot")}
            autoResize
            rows={6}
          />
        </div>
      )}

      {step === 2 && (
        <div className="space-y-4">
          <Input label="Language" {...register("language")} />
          <Input
            label="Play Online URL"
            type="url"
            placeholder="https://"
            {...register("play_online_url")}
            error={errors.play_online_url?.message}
          />
          <div className="flex items-center gap-2">
            <Checkbox
              id="is_free"
              checked={formData.is_free}
              onCheckedChange={(checked) =>
                setValue("is_free", checked === true)
              }
            />
            <Label htmlFor="is_free">This game is free</Label>
          </div>
          <div className="flex items-center gap-2">
            <Checkbox
              id="has_purchasable_content"
              checked={formData.has_purchasable_content}
              onCheckedChange={(checked) =>
                setValue("has_purchasable_content", checked === true)
              }
            />
            <Label htmlFor="has_purchasable_content">
              Has purchasable content
            </Label>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="space-y-3 rounded-lg border p-4 text-sm">
          <h3 className="font-semibold">Review your submission</h3>
          <dl className="grid gap-2">
            <ReviewItem label="Title" value={formData.title} />
            <ReviewItem
              label="Engine"
              value={
                engines.find((e) => String(e.id) === formData.engine_id)?.name
              }
            />
            <ReviewItem label="Status" value={formData.development_status} />
            <ReviewItem label="Rating" value={formData.rating} />
            <ReviewItem label="Language" value={formData.language} />
            <ReviewItem
              label="Free"
              value={formData.is_free ? "Yes" : "No"}
            />
          </dl>
        </div>
      )}

      <div className="flex justify-between">
        <Button
          type="button"
          variant="outline"
          onClick={prevStep}
          disabled={step === 0}
        >
          <ChevronLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        {step < STEPS.length - 1 ? (
          <Button type="button" onClick={nextStep}>
            Next
            <ChevronRight className="ml-2 h-4 w-4" />
          </Button>
        ) : (
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Submitting..." : "Submit Game"}
          </Button>
        )}
      </div>
    </form>
  );
}

function ReviewItem({
  label,
  value,
}: {
  label: string;
  value?: string;
}) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium capitalize">{value || "—"}</dd>
    </div>
  );
}
