import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

function num(params: Record<string, ParamValue>, key: string, def: number): number {
  const v = params[key];
  if (v == null) return def;
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param ${key}`);
}

function elementwiseMerge(name: string): void {
  registerBlock({
    name,
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error(`${name} requires inputs`);
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });
}

export function registerMerge(): void {
  elementwiseMerge("Add");
  elementwiseMerge("Mul");
  elementwiseMerge("Sub");
  elementwiseMerge("Div");

  // Concat(axis=1): concatenate shapes along axis
  registerBlock({
    name: "Concat",
    params: [{ name: "axis", type: "number", required: false }],
    inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("Concat requires inputs");
      const axis = num(params, "axis", 1);
      const out = [...inputs[0]];
      for (let i = 1; i < inputs.length; i++) {
        out[axis] = out[axis] + inputs[i][axis];
      }
      return [out];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // MatMul: (A,B) x (B,C) → (A,C)
  registerBlock({
    name: "MatMul",
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length < 2) throw new Error("MatMul requires two inputs");
      const [A] = inputs[0];
      const [, C] = inputs[1];
      return [[A, C]];
    },
    paramCount: () => 0,
    showDepth: false,
  });
}
