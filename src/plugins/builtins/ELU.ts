import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

const ELU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "alpha", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("ELU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const alpha = getNum(block.params, "alpha") ?? 1.0;
      return {
        init: `self.${block.id} = nn.ELU(alpha=${alpha})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(block, inputVars, outputVars) {
      const alpha = getNum(block.params, "alpha") ?? 1.0;
      return {
        forward: `${outputVars[0]} = keras.layers.ELU(alpha=${alpha})(${inputVars[0]})`,
      };
    },
    candle(block, inputVars, outputVars) {
      const alpha = getNum(block.params, "alpha") ?? 1.0;
      return {
        forward: `${outputVars[0]} = ${inputVars[0]}.elu(${alpha})?;`,
      };
    },
  },
};

export default ELU;
