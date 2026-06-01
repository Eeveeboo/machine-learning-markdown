import type { BlockPlugin } from "../types.js";

/** @codegen candle: placeholder — learnable PReLU slopes not supported in candle_nn */
const PReLU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("PReLU requires an input");
    return [inputs[0]];
  },
  paramCount(inputs, _params) {
    if (inputs.length === 0 || inputs[0].length === 0) return 0;
    return inputs[0][0]; // one slope per input channel
  },
  codegen: {
    pytorch(block, inputVars) {
      return {
        attr: { name: block.id, init: "nn.PReLU()" },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(block, inputVars) {
      return {
        attr: null,
        forward: `keras.layers.PReLU()(${inputVars[0] ?? "x"})`,
      };
    },
    candle(_block, inputVars) {
      // candle-core does not provide a built-in PReLU; emit a comment placeholder
      const x = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `${x}.relu()?  /* PReLU: learnable slopes not supported in candle codegen */`,
      };
    },
  },
};

export default PReLU;
