import type { BlockPlugin } from "../types.js";

const Map: BlockPlugin = {
  inputs: ["x"],
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Map requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      return { attr: null, forward: `${x}  # Map (identity; apply custom fn manually)` };
    },
    keras(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      return { attr: null, forward: `keras.layers.Lambda(lambda t: t)(${x})  # Map` };
    },
    candle(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      return { attr: null, forward: `${x}.clone()  // Map (identity; apply custom fn manually)` };
    },
  },
};

export default Map;
