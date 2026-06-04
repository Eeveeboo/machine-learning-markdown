import type { BlockPlugin } from "../types.js";

const ReLU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("ReLU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      return {
        init: `self.${block.id} = nn.ReLU()`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.ReLU()(${inputVars[0]})`,
      };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}.relu()?;` };
    },
  },
};

export default ReLU;
