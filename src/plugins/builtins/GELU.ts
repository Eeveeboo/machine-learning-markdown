import type { BlockPlugin } from "../types.js";

const GELU: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("GELU requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      return {
        init: `self.${block.id} = nn.GELU()`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(_block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.Activation('gelu')(${inputVars[0]})`,
      };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}.gelu()?` };
    },
  },
};

export default GELU;
