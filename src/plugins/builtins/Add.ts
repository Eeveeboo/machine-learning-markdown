import type { BlockPlugin } from "../types.js";

const Add: BlockPlugin = {
  inputs: ["a", "b"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Add requires inputs");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars.join(" + ")}` };
    },
    keras(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = keras.layers.Add()([${inputVars.join(", ")}])` };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = (&${inputVars[0]} + &${inputVars[1]})?;` };
    },
  },
};

export default Add;
