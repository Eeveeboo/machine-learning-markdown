import type { BlockPlugin } from "../types.js";

const Flatten: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Flatten requires an input");
    const flat = inputs[0].reduce((a, b) => a * b, 1);
    return [[flat]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      return {
        attr: { name: block.id, init: "nn.Flatten()" },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(_block, inputVars) {
      return {
        attr: null,
        forward: `keras.layers.Flatten()(${inputVars[0] ?? "x"})`,
      };
    },
    candle(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      return { attr: null, forward: `${x}.flatten_from(1)?` };
    },
  },
};

export default Flatten;
