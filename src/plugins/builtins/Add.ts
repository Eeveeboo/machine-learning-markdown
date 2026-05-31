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
    pytorch(_block, inputVars) {
      return { attr: null, forward: `${inputVars.join(" + ")}` };
    },
    keras(_block, inputVars) {
      return { attr: null, forward: `keras.layers.Add()([${inputVars.join(", ")}])` };
    },
    candle(_block, inputVars) {
      const a = inputVars[0] ?? "x";
      const b = inputVars[1] ?? "x";
      return { attr: null, forward: `(&${a} + &${b})?` };
    },
  },
};

export default Add;
