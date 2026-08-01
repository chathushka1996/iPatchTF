"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

export interface SliderProps
  extends Omit<
    React.InputHTMLAttributes<HTMLInputElement>,
    "type" | "value" | "defaultValue" | "onChange"
  > {
  value?: number;
  defaultValue?: number;
  min?: number;
  max?: number;
  step?: number;
  onValueChange?: (value: number) => void;
  label?: string;
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      className,
      value,
      defaultValue = 0,
      min = 0,
      max = 100,
      step = 1,
      onValueChange,
      label,
      id,
      ...props
    },
    ref,
  ) => {
    const sliderId = id || React.useId();
    const [internalValue, setInternalValue] = React.useState(
      value ?? defaultValue,
    );
    const currentValue = value ?? internalValue;
    const percentage = ((currentValue - min) / (max - min)) * 100;

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const next = Number(e.target.value);
      if (value === undefined) setInternalValue(next);
      onValueChange?.(next);
    };

    return (
      <div className={cn("space-y-2", className)}>
        {label && (
          <div className="flex items-center justify-between">
            <label htmlFor={sliderId} className="text-sm font-medium">
              {label}
            </label>
            <span className="text-sm text-muted-foreground">{currentValue}</span>
          </div>
        )}
        <div className="relative flex w-full items-center">
          <div className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
            <div
              className="absolute h-full bg-primary transition-all"
              style={{ width: `${percentage}%` }}
            />
          </div>
          <input
            ref={ref}
            id={sliderId}
            type="range"
            min={min}
            max={max}
            step={step}
            value={currentValue}
            onChange={handleChange}
            className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
            {...props}
          />
        </div>
      </div>
    );
  },
);
Slider.displayName = "Slider";

export { Slider };
