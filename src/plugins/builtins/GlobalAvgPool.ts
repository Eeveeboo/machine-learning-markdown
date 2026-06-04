import type { BlockPlugin } from "../types.js";

export const GlobalAvgPool: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("GlobalAvgPool requires an input");
    const [C] = inputs[0];
    return [[C, 1, 1]];
  },

  paramCount() {
    return 0;
  },

  codegen: {
    pytorch(block, inputVars, outputVars) {
      return {
        init: `self.${block.id} = nn.AdaptiveAvgPool2d(1)`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(_block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.GlobalAveragePooling2D()(${inputVars[0]})`,
      };
    },

    candle(_block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = ${inputVars[0]}.mean_keepdim(candle_core::D::Minus1)?.mean_keepdim(candle_core::D::Minus2)?;`,
      };
    },
  },
};

export default GlobalAvgPool;
