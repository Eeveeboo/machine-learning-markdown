import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

function num(params: Record<string, ParamValue>, key: string, def: number): number {
  const v = params[key];
  if (v == null) return def;
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param ${key}`);
}

export function registerRecurrent(): void {
  // LSTM: (seq, input) → outputShapes[0]=(seq, H), outputShapes[1]=(H,)
  registerBlock({
    name: "LSTM",
    params: [
      { name: "hidden_size", type: "number", required: true },
      { name: "num_layers", type: "number", required: false },
    ],
    inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("LSTM requires an input");
      const [seq] = inputs[0];
      const H = num(params, "hidden_size", NaN);
      return [[seq, H], [H]];
    },
    paramCount(inputs, params) {
      const inSize = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
      const H = num(params, "hidden_size", NaN);
      const L = num(params, "num_layers", 1);
      // Per layer: 4 gates × (input*H + H*H + H + H) = 4*(in*H + H² + 2H)
      // Subsequent layers use H as input: 4*(H*H + H*H + 2H) = 8*H² + 8*H
      const firstLayer = 4 * (inSize * H + H * H + 2 * H);
      const extraLayer = Math.max(0, L - 1) * 4 * (H * H + H * H + 2 * H);
      return firstLayer + extraLayer;
    },
  });

  // GRU: (seq, input) → outputShapes[0]=(seq, H), outputShapes[1]=(H,)
  registerBlock({
    name: "GRU",
    params: [
      { name: "hidden_size", type: "number", required: true },
      { name: "num_layers", type: "number", required: false },
    ],
    inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("GRU requires an input");
      const [seq] = inputs[0];
      const H = num(params, "hidden_size", NaN);
      return [[seq, H], [H]];
    },
    paramCount(inputs, params) {
      const inSize = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
      const H = num(params, "hidden_size", NaN);
      const L = num(params, "num_layers", 1);
      // GRU has 3 gates
      const firstLayer = 3 * (inSize * H + H * H + 2 * H);
      const extraLayer = Math.max(0, L - 1) * 3 * (H * H + H * H + 2 * H);
      return firstLayer + extraLayer;
    },
  });

  // RNN: (seq, input) → (seq, H)
  registerBlock({
    name: "RNN",
    params: [
      { name: "hidden_size", type: "number", required: true },
      { name: "num_layers", type: "number", required: false },
    ],
    inferShape(inputs: Shape[], params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("RNN requires an input");
      const [seq] = inputs[0];
      const H = num(params, "hidden_size", NaN);
      return [[seq, H]];
    },
    paramCount(inputs, params) {
      const inSize = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
      const H = num(params, "hidden_size", NaN);
      const L = num(params, "num_layers", 1);
      // RNN has 1 gate
      const firstLayer = 1 * (inSize * H + H * H + 2 * H);
      const extraLayer = Math.max(0, L - 1) * 1 * (H * H + H * H + 2 * H);
      return firstLayer + extraLayer;
    },
  });
}
