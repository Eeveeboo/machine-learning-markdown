import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

const Softmax: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Softmax requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      const dim = getNum(block.params, "dim") ?? -1;
      return {
        attr: { name: block.id, init: `nn.Softmax(dim=${dim})` },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(block, inputVars) {
      const axis = getNum(block.params, "dim") ?? -1;
      return {
        attr: null,
        forward: `keras.layers.Softmax(axis=${axis})(${inputVars[0] ?? "x"})`,
      };
    },
    candle(block, inputVars) {
      const dim = getNum(block.params, "dim") ?? -1;
      const x = inputVars[0] ?? "x";
      // candle uses softmax with explicit dim; -1 maps to last dim
      return {
        attr: null,
        forward: `candle_nn::ops::softmax(&${x}, candle_core::D::Minus${dim === -1 ? 1 : Math.abs(dim)})?`,
      };
    },
  },
};

export default Softmax;
