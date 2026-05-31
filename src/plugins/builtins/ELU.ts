import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

const ELU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("ELU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      const alpha = getNum(block.params, "alpha") ?? 1.0;
      return {
        attr: { name: block.id, init: `nn.ELU(alpha=${alpha})` },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(block, inputVars) {
      const alpha = getNum(block.params, "alpha") ?? 1.0;
      return {
        attr: null,
        forward: `keras.layers.ELU(alpha=${alpha})(${inputVars[0] ?? "x"})`,
      };
    },
    candle(block, inputVars) {
      const alpha = getNum(block.params, "alpha") ?? 1.0;
      const x = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `${x}.elu(${alpha})?`,
      };
    },
  },
};

export default ELU;
