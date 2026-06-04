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
    pytorch(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const normalized = inputShape.length ? `[${inputShape.slice(1).join(", ")}]` : "[]";
      return {
        init: `self.${block.id} = nn.LayerNorm(${normalized})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.LayerNormalization()(${inputVars[0]})`,
      };
    },

    candle(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const features = inputShape[inputShape.length - 1] ?? 0;
      return {
        init: {
          field: `${block.id}: candle_nn::LayerNorm`,
          body: `let ${block.id} = candle_nn::layer_norm(${features}, 1e-5, vb.pp("${block.id}"))?;`,
        },
        forward: `${outputVars[0]} = self.${block.id}.forward(&${inputVars[0]})?`,
      };
    },
  },
};

export default LayerNorm;
