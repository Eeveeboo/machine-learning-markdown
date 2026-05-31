/**
 * Shared helper utilities for builtin block plugins.
 */

import type { ParamValue } from "../../ast/nodes.js";

// ---------------------------------------------------------------------------
// Param accessors
// ---------------------------------------------------------------------------

/**
 * Safely get a numeric param value. Returns `defaultVal` (or `undefined`) if
 * the param is absent or not a number.
 */
export function getNum(
  params: Record<string, ParamValue>,
  key: string,
  defaultVal?: number,
): number | undefined {
  const v = params[key];
  if (v == null) return defaultVal;
  if (v.kind === "number") return v.value;
  return defaultVal;
}

/**
 * Safely get a string param value. Accepts both `"string"` and `"bareword"`
 * kinds. Returns `defaultVal` (or `undefined`) if absent.
 */
export function getStr(
  params: Record<string, ParamValue>,
  key: string,
  defaultVal?: string,
): string | undefined {
  const v = params[key];
  if (v == null) return defaultVal;
  if (v.kind === "string" || v.kind === "bareword") return v.value;
  return defaultVal;
}

/**
 * Get a required numeric param, throwing if absent or wrong type.
 */
export function requireNum(
  params: Record<string, ParamValue>,
  key: string,
): number {
  const v = params[key];
  if (v == null) throw new Error(`Missing required numeric param "${key}"`);
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param "${key}", got "${v.kind}"`);
}

/**
 * Get a list of numbers from a `shape` or `list` param.
 * For a `shape` param, returns its dims.
 * For a `list` param, returns only the numeric items.
 * Returns an empty array if the param is absent.
 */
export function getNumList(
  params: Record<string, ParamValue>,
  key: string,
): number[] {
  const v = params[key];
  if (v == null) return [];
  if (v.kind === "shape") return v.dims;
  if (v.kind === "list") {
    return v.items
      .filter((i): i is Extract<ParamValue, { kind: "number" }> => i.kind === "number")
      .map((i) => i.value);
  }
  return [];
}

/**
 * Get a required shape param (throws if absent or not a shape).
 */
export function requireShape(
  params: Record<string, ParamValue>,
  key: string,
): number[] {
  const v = params[key];
  if (v == null) throw new Error(`Missing required shape param "${key}"`);
  if (v.kind === "shape") return v.dims;
  throw new Error(`Expected shape for param "${key}", got "${v.kind}"`);
}

// ---------------------------------------------------------------------------
// Shape arithmetic
// ---------------------------------------------------------------------------

/**
 * Compute the output size of a convolution dimension.
 *
 * Formula: floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
 *
 * Matches PyTorch's Conv1d/Conv2d/Conv3d semantics.
 */
export function convOutputSize(
  input: number,
  kernel: number,
  padding: number,
  stride: number,
  dilation = 1,
): number {
  return Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
}

/**
 * Simplified convolution output size (dilation=1).
 * Equivalent to the original `convOut` used in layers.ts.
 */
export function convOut(
  size: number,
  kernel: number,
  stride: number,
  padding: number,
): number {
  return convOutputSize(size, kernel, padding, stride, 1);
}

/**
 * Compute the output size of a transposed convolution dimension.
 *
 * Formula: (input - 1) * stride - 2 * padding + kernel
 */
export function convTransposeOutputSize(
  input: number,
  kernel: number,
  padding: number,
  stride: number,
): number {
  return (input - 1) * stride - 2 * padding + kernel;
}
