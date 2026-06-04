import type { BlockPlugin } from "../types.js";

const Flatten: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Flatten requires an input");
    const flat = inputs[0].reduce((a, b) => a * b, 1);
    return [[flat]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      return {
        init: `self.${block.id} = nn.Flatten()`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(_block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.Flatten()(${inputVars[0]})`,
      };
    },
    candle(_block, inputVars, outputVars) {
      return { forward: `${outputVars[0]} = ${inputVars[0]}.flatten_from(1)?;` };
    },
  },
};

export default Flatten;
