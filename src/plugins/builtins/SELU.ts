import type { BlockPlugin } from "../types.js";

const SELU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("SELU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      return {
        init: `self.${block.id} = nn.SELU()`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(_block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.Activation('selu')(${inputVars[0]})`,
      };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}.selu()?;` };
    },
  },
};

export default SELU;
