import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

function num(params: Record<string, ParamValue>, key: string, def: number): number {
  const v = params[key];
  if (v == null) return def;
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param ${key}`);
}

export function registerExtra(): void {
  // Split(chunks=N, axis=0): (S,...) → N copies of (S/N,...) on that axis
  registerBlock({
    name: "Split",
    params: [
      { name: "chunks", type: "number", required: true },
      { name: "axis", type: "number", required: false },
    ],
    inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Split requires an input");
      const N = num(params, "chunks", NaN);
      const axis = num(params, "axis", 0);
      const outShape = [...inputs[0]];
      outShape[axis] = Math.floor(inputs[0][axis] / N);
      return Array.from({ length: N }, () => [...outShape]);
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // Repeat: passthrough
  registerBlock({
    name: "Repeat",
    params: [{ name: "times", type: "number", required: false }],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Repeat requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // Map: passthrough
  registerBlock({
    name: "Map",
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Map requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // Gather: passthrough
  registerBlock({
    name: "Gather",
    params: [{ name: "axis", type: "number", required: false }],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Gather requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });
}
