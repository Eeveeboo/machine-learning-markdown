import type { BlockPlugin } from "../types.js";

export const LayerNorm: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("LayerNorm requires an input");
    return [inputs[0]];
  },

  paramCount(inputs, _params) {
    if (inputs.length === 0 || inputs[0].length === 0) return 0;
    return inputs[0][inputs[0].length - 1] * 2;
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const normalized = inputShape.length ? `[${inputShape.slice(1).join(", ")}]` : "[]";
      return {
        attr: { name: block.id, init: `nn.LayerNorm(${normalized})` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.LayerNormalization()(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const features = inputShape[inputShape.length - 1] ?? 0;
      return {
        attr: {
          name: block.id,
          init: "",
          typeAnnotation: "candle_nn::LayerNorm",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default LayerNorm;
