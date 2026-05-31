import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

function num(params: Record<string, ParamValue>, key: string, def: number): number {
  const v = params[key];
  if (v == null) return def;
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param ${key}`);
}

function shapeParam(params: Record<string, ParamValue>, key: string): number[] | null {
  const v = params[key];
  if (v == null) return null;
  if (v.kind === "shape") return v.dims;
  if (v.kind === "list") {
    return v.items.map(i => {
      if (i.kind === "number") return i.value;
      throw new Error(`Expected number in list for ${key}`);
    });
  }
  return null;
}

function poolOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

function poolBlock(
  name: string,
  params: { name: string; type: "number" | "string" | "bool" | "shape" | "list" | "bareword"; required: boolean }[],
  infer: (inputs: Shape[], params: Record<string, ParamValue>) => Shape[]
): void {
  registerBlock({
    name,
    params,
    inferShape: infer,
    paramCount: () => 0,
    showDepth: false,
  });
}

export function registerPool(): void {
  // MaxPool
  poolBlock("MaxPool", [
    { name: "kernel", type: "number", required: true },
    { name: "stride", type: "number", required: false },
    { name: "padding", type: "number", required: false },
  ], (inputs, params) => {
    if (inputs.length === 0) throw new Error("MaxPool requires an input");
    const [C, H, W] = inputs[0];
    const K = num(params, "kernel", NaN);
    const S = num(params, "stride", 1);
    const P = num(params, "padding", 0);
    return [[C, poolOut(H, K, S, P), poolOut(W, K, S, P)]];
  });

  // AvgPool
  poolBlock("AvgPool", [
    { name: "kernel", type: "number", required: true },
    { name: "stride", type: "number", required: false },
    { name: "padding", type: "number", required: false },
  ], (inputs, params) => {
    if (inputs.length === 0) throw new Error("AvgPool requires an input");
    const [C, H, W] = inputs[0];
    const K = num(params, "kernel", NaN);
    const S = num(params, "stride", 1);
    const P = num(params, "padding", 0);
    return [[C, poolOut(H, K, S, P), poolOut(W, K, S, P)]];
  });

  // GlobalAvgPool
  poolBlock("GlobalAvgPool", [], (inputs, _params) => {
    if (inputs.length === 0) throw new Error("GlobalAvgPool requires an input");
    const [C] = inputs[0];
    return [[C, 1, 1]];
  });

  // AdaptiveAvgPool
  poolBlock("AdaptiveAvgPool", [{ name: "size", type: "shape", required: true }], (inputs, params) => {
    if (inputs.length === 0) throw new Error("AdaptiveAvgPool requires an input");
    const [C] = inputs[0];
    const size = shapeParam(params, "size");
    if (!size || size.length < 2) throw new Error("AdaptiveAvgPool requires size=(H,W)");
    return [[C, size[0], size[1]]];
  });
}
