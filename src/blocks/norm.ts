import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

export function registerNorm(): void {
  // BatchNorm(num_features) — 2 params per feature (gamma + beta)
  registerBlock({
    name: "BatchNorm",
    params: [{ name: "num_features", type: "number", required: false }],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("BatchNorm requires an input");
      return [inputs[0]];
    },
    paramCount(inputs, params) {
      const nf = params["num_features"]?.kind === "number" ? params["num_features"].value : (inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0);
      return nf * 2;
    },
  });

  // LayerNorm — 2 params per feature (gamma + beta), feature dim = last dim
  registerBlock({
    name: "LayerNorm",
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("LayerNorm requires an input");
      return [inputs[0]];
    },
    paramCount(inputs, _params) {
      if (inputs.length === 0 || inputs[0].length === 0) return 0;
      return inputs[0][inputs[0].length - 1] * 2;
    },
    showDepth: false,
  });

  // GroupNorm(num_groups) — 2 params per channel (gamma + beta)
  registerBlock({
    name: "GroupNorm",
    params: [{ name: "num_groups", type: "number", required: false }],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("GroupNorm requires an input");
      return [inputs[0]];
    },
    paramCount(inputs, _params) {
      if (inputs.length === 0 || inputs[0].length === 0) return 0;
      return inputs[0][0] * 2; // C * 2
    },
  });

  // InstanceNorm — 2 params per channel (gamma + beta)
  registerBlock({
    name: "InstanceNorm",
    params: [],
    inferShape(inputs: Shape[], _params: Record<string, ParamValue>): Shape[] {
      if (inputs.length === 0) throw new Error("InstanceNorm requires an input");
      return [inputs[0]];
    },
    paramCount(inputs, _params) {
      if (inputs.length === 0 || inputs[0].length === 0) return 0;
      return inputs[0][0] * 2; // C * 2
    },
    showDepth: false,
  });
}
