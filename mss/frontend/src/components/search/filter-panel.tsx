"use client";

import { useState } from "react";
import { Search } from "lucide-react";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  CONTENT_RATINGS,
  DEVELOPMENT_STATUSES,
  PC_GENDERS,
} from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Engine } from "@/types/game";
import type { GameFilters } from "@/types/search";

interface FilterOption {
  value: string;
  label: string;
}

interface FilterPanelProps {
  filters: GameFilters;
  onChange: (filters: GameFilters) => void;
  engines?: Engine[];
  genres?: FilterOption[];
  adultThemes?: FilterOption[];
  transformations?: FilterOption[];
  multimedia?: FilterOption[];
  contentWarnings?: FilterOption[];
  className?: string;
}

export function FilterPanel({
  filters,
  onChange,
  engines = [],
  genres = [],
  adultThemes = [],
  transformations = [],
  multimedia = [],
  contentWarnings = [],
  className,
}: FilterPanelProps) {
  const [engineSearch, setEngineSearch] = useState("");

  const toggleArrayFilter = (
    key: keyof GameFilters,
    value: string,
  ) => {
    const current = (filters[key] as string[] | undefined) ?? [];
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    onChange({ ...filters, [key]: next.length > 0 ? next : undefined });
  };

  const filteredEngines = engines.filter((e) =>
    e.name.toLowerCase().includes(engineSearch.toLowerCase()),
  );

  return (
    <div className={cn("space-y-2", className)}>
      <Accordion type="multiple" defaultValue={["status", "engine"]}>
        <AccordionItem value="status">
          <AccordionTrigger>Development Status</AccordionTrigger>
          <AccordionContent>
            <CheckboxGroup
              options={DEVELOPMENT_STATUSES.map((s) => ({
                value: s.value,
                label: s.label,
              }))}
              selected={filters.status ?? []}
              onToggle={(v) => toggleArrayFilter("status", v)}
            />
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="engine">
          <AccordionTrigger>Engine</AccordionTrigger>
          <AccordionContent className="space-y-3">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search engines..."
                value={engineSearch}
                onChange={(e) => setEngineSearch(e.target.value)}
                className="h-8 pl-8 text-sm"
              />
            </div>
            <CheckboxGroup
              options={filteredEngines.map((e) => ({
                value: e.slug,
                label: e.name,
              }))}
              selected={filters.engine ?? []}
              onToggle={(v) => toggleArrayFilter("engine", v)}
            />
          </AccordionContent>
        </AccordionItem>

        {genres.length > 0 && (
          <AccordionItem value="genre">
            <AccordionTrigger>Genre</AccordionTrigger>
            <AccordionContent>
              <CheckboxGroup
                options={genres}
                selected={filters.genre ?? []}
                onToggle={(v) => toggleArrayFilter("genre", v)}
              />
            </AccordionContent>
          </AccordionItem>
        )}

        {adultThemes.length > 0 && (
          <AccordionItem value="adult_theme">
            <AccordionTrigger>Adult Themes</AccordionTrigger>
            <AccordionContent>
              <CheckboxGroup
                options={adultThemes}
                selected={filters.adult_theme ?? []}
                onToggle={(v) => toggleArrayFilter("adult_theme", v)}
              />
            </AccordionContent>
          </AccordionItem>
        )}

        {transformations.length > 0 && (
          <AccordionItem value="transformation">
            <AccordionTrigger>Transformation</AccordionTrigger>
            <AccordionContent>
              <CheckboxGroup
                options={transformations}
                selected={filters.transformation ?? []}
                onToggle={(v) => toggleArrayFilter("transformation", v)}
              />
            </AccordionContent>
          </AccordionItem>
        )}

        {multimedia.length > 0 && (
          <AccordionItem value="multimedia">
            <AccordionTrigger>Multimedia</AccordionTrigger>
            <AccordionContent>
              <CheckboxGroup
                options={multimedia}
                selected={filters.multimedia ?? []}
                onToggle={(v) => toggleArrayFilter("multimedia", v)}
              />
            </AccordionContent>
          </AccordionItem>
        )}

        {contentWarnings.length > 0 && (
          <AccordionItem value="content_warning">
            <AccordionTrigger>Content Warnings</AccordionTrigger>
            <AccordionContent>
              <CheckboxGroup
                options={contentWarnings}
                selected={filters.content_warning ?? []}
                onToggle={(v) => toggleArrayFilter("content_warning", v)}
              />
            </AccordionContent>
          </AccordionItem>
        )}

        <AccordionItem value="rating">
          <AccordionTrigger>Content Rating</AccordionTrigger>
          <AccordionContent>
            <CheckboxGroup
              options={CONTENT_RATINGS.map((r) => ({
                value: r.value,
                label: r.label,
              }))}
              selected={filters.rating ?? []}
              onToggle={(v) => toggleArrayFilter("rating", v)}
            />
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="pc_gender">
          <AccordionTrigger>PC Gender</AccordionTrigger>
          <AccordionContent>
            <CheckboxGroup
              options={PC_GENDERS.map((g) => ({
                value: g.value,
                label: g.label,
              }))}
              selected={filters.pc_gender ?? []}
              onToggle={(v) => toggleArrayFilter("pc_gender", v)}
            />
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="likes">
          <AccordionTrigger>Likes</AccordionTrigger>
          <AccordionContent>
            <Slider
              label="Minimum likes"
              min={0}
              max={1000}
              step={10}
              value={filters.min_likes ?? 0}
              onValueChange={(v) =>
                onChange({ ...filters, min_likes: v > 0 ? v : undefined })
              }
            />
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="play_online">
          <AccordionTrigger>Play Online</AccordionTrigger>
          <AccordionContent>
            <div className="flex items-center justify-between">
              <Label htmlFor="play-online">Has online play</Label>
              <Switch
                id="play-online"
                checked={filters.has_play_online ?? false}
                onCheckedChange={(checked) =>
                  onChange({
                    ...filters,
                    has_play_online: checked || undefined,
                  })
                }
              />
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="author">
          <AccordionTrigger>Author</AccordionTrigger>
          <AccordionContent>
            <Input
              placeholder="Search by author..."
              value={filters.author ?? ""}
              onChange={(e) =>
                onChange({
                  ...filters,
                  author: e.target.value || undefined,
                })
              }
            />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}

function CheckboxGroup({
  options,
  selected,
  onToggle,
}: {
  options: FilterOption[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  return (
    <div className="max-h-48 space-y-2 overflow-y-auto scrollbar-thin">
      {options.map((option) => (
        <div key={option.value} className="flex items-center gap-2">
          <Checkbox
            id={`filter-${option.value}`}
            checked={selected.includes(option.value)}
            onCheckedChange={() => onToggle(option.value)}
          />
          <Label
            htmlFor={`filter-${option.value}`}
            className="cursor-pointer text-sm font-normal"
          >
            {option.label}
          </Label>
        </div>
      ))}
    </div>
  );
}
