import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

const LeakyReLU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "negative_slope", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("LeakyReLU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const slope = getNum(block.params, "negative_slope") ?? 0.01;
      return {
        init: `self.${block.id} = nn.LeakyReLU(${slope})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(block, inputVars, outputVars) {
      const slope = getNum(block.params, "negative_slope") ?? 0.01;
      return {
        forward: `${outputVars[0]} = keras.layers.LeakyReLU(alpha=${slope})(${inputVars[0]})`,
      };
    },
    candle(block, inputVars, outputVars) {
      const slope = getNum(block.params, "negative_slope") ?? 0.01;
      return {
        forward: `${outputVars[0]} = ${inputVars[0]}.leaky_relu(${slope})?;`,
      };
    },
  },
};

export default LeakyReLU;
