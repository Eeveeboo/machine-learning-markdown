import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

function passthrough(name: string): void {
  registerBlock({
    name,
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error(`${name} requires an input`);
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });
}

function num(params: Record<string, ParamValue>, key: string, def: number): number {
  const v = params[key];
  if (v == null) return def;
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param ${key}`);
}

export function registerActivations(): void {
  passthrough("ReLU");
  passthrough("SELU");
  passthrough("GELU");
  passthrough("Sigmoid");
  passthrough("Tanh");

  // Softmax has an optional axis param but output shape is same
  registerBlock({
    name: "Softmax",
    params: [{ name: "axis", type: "number", required: false }],
    inferShape(inputs, _params) {
      if (inputs.length === 0) throw new Error("Softmax requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // LeakyReLU has optional negative_slope
  registerBlock({
    name: "LeakyReLU",
    params: [{ name: "negative_slope", type: "number", required: false }],
    inferShape(inputs, _params) {
      if (inputs.length === 0) throw new Error("LeakyReLU requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
    showDepth: false,
  });

  // PReLU has optional num_parameters
  registerBlock({
    name: "PReLU",
    params: [{ name: "num_parameters", type: "number", required: false }],
    inferShape(inputs, _params) {
      if (inputs.length === 0) throw new Error("PReLU requires an input");
      return [inputs[0]];
    },
    paramCount(inputs, _params) {
      if (inputs.length === 0 || inputs[0].length === 0) return 0;
      return inputs[0][0]; // one slope per input channel
    },
    showDepth: false,
  });
}
