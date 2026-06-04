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
    pytorch(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}  # Map (identity; apply custom fn manually)` };
    },
    keras(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = keras.layers.Lambda(lambda t: t)(${inputVars[0]})  # Map` };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}.clone()  // Map (identity; apply custom fn manually)` };
    },
  },
};

export default Map;
