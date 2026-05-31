import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

const LeakyReLU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("LeakyReLU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      const slope = getNum(block.params, "negative_slope") ?? 0.01;
      return {
        attr: { name: block.id, init: `nn.LeakyReLU(${slope})` },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(block, inputVars) {
      const slope = getNum(block.params, "negative_slope") ?? 0.01;
      return {
        attr: null,
        forward: `keras.layers.LeakyReLU(alpha=${slope})(${inputVars[0] ?? "x"})`,
      };
    },
    candle(block, inputVars) {
      const slope = getNum(block.params, "negative_slope") ?? 0.01;
      const x = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `${x}.leaky_relu(${slope})?`,
      };
    },
  },
};

export default LeakyReLU;
