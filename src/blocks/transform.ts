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

export function registerTransform(): void {
  // Flatten: (C,H,W) → (C*H*W,)
  registerBlock({
    name: "Flatten",
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Flatten requires an input");
      const flat = inputs[0].reduce((a, b) => a * b, 1);
      return [[flat]];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // Reshape(shape) → shape
  registerBlock({
    name: "Reshape",
    params: [{ name: "shape", type: "shape", required: true }],
    inferShape(_inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      const s = shapeParam(params, "shape");
      if (!s) throw new Error("Reshape requires shape param");
      return [s];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // Dropout: passthrough
  registerBlock({
    name: "Dropout",
    params: [{ name: "rate", type: "number", required: false }],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Dropout requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // Pad(padding=(P0,P1,P2,P3)): (C,H,W) → (C, H+P0+P1, W+P2+P3)
  registerBlock({
    name: "Pad",
    params: [{ name: "padding", type: "shape", required: true }],
    inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Pad requires an input");
      const [C, H, W] = inputs[0];
      const p = shapeParam(params, "padding");
      if (!p || p.length < 4) throw new Error("Pad requires padding=(P0,P1,P2,P3)");
      return [[C, H + p[0] + p[1], W + p[2] + p[3]]];
    },
    paramCount: () => 0,
    showDepth: false,
  });
}
