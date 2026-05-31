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
    pytorch(block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: { name: block.id, init: `nn.AdaptiveAvgPool2d(1)` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(_block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.GlobalAveragePooling2D()(${mainIn})`,
      };
    },

    candle(_block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `${mainIn}.mean_keepdim(candle_core::D::Minus1)?.mean_keepdim(candle_core::D::Minus2)?`,
      };
    },
  },
};

export default GlobalAvgPool;
