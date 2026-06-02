import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

export const GroupNorm: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "num_groups", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("GroupNorm requires an input");
    return [inputs[0]];
  },

  paramCount(inputs, _params) {
    if (inputs.length === 0 || inputs[0].length === 0) return 0;
    return inputs[0][0] * 2; // C * 2
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const groups = getNum(block.params, "num_groups") ?? 32;
      const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      return {
        attr: { name: block.id, init: `nn.GroupNorm(${groups}, ${num})` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      const groups = getNum(block.params, "num_groups") ?? 32;
      return {
        attr: null,
        forward: `keras.layers.GroupNormalization(groups=${groups})(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const groups = getNum(block.params, "num_groups") ?? 32;
      const channels = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      return {
        attr: {
          name: block.id,
          init: `candle_nn::group_norm(${groups}, ${channels}, 1e-5, vb.pp("${block.id}"))?`,
          typeAnnotation: "candle_nn::GroupNorm",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default GroupNorm;
