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
    pytorch(block, inputVars, outputVars) {
      return {
        init: `self.${block.id} = nn.PReLU()`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.PReLU()(${inputVars[0]})`,
      };
    },
    candle(_block, inputVars, outputVars) {
      // candle-core does not provide a built-in PReLU; emit a comment placeholder
      return {
        forward: `${outputVars[0]} = ${inputVars[0]}.relu()?;  /* PReLU: learnable slopes not supported in candle codegen */`,
      };
    },
  },
};

export default PReLU;
