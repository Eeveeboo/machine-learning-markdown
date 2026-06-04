import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

const Softmax: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "dim", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Softmax requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const dim = getNum(block.params, "dim") ?? -1;
      return {
        init: `self.${block.id} = nn.Softmax(dim=${dim})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(block, inputVars, outputVars) {
      const axis = getNum(block.params, "dim") ?? -1;
      return {
        forward: `${outputVars[0]} = keras.layers.Softmax(axis=${axis})(${inputVars[0]})`,
      };
    },
    candle(block, inputVars, outputVars) {
      const dim = getNum(block.params, "dim") ?? -1;
      // candle uses softmax with explicit dim; -1 maps to last dim
      return {
        forward: `${outputVars[0]} = candle_nn::ops::softmax(&${inputVars[0]}, candle_core::D::Minus${dim === -1 ? 1 : Math.abs(dim)})?`,
      };
    },
  },
};

export default Softmax;
