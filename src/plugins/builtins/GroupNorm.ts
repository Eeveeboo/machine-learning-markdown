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
    pytorch(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const groups = getNum(block.params, "num_groups") ?? 32;
      const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      return {
        init: `self.${block.id} = nn.GroupNorm(${groups}, ${num})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(block, inputVars, outputVars) {
      const groups = getNum(block.params, "num_groups") ?? 32;
      return {
        init: null,
        forward: `${outputVars[0]} = keras.layers.GroupNormalization(groups=${groups})(${inputVars[0]})`,
      };
    },

    candle(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const groups = getNum(block.params, "num_groups") ?? 32;
      const channels = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      return {
        init: {
          field: `${block.id}: candle_nn::GroupNorm`,
          body: `let ${block.id} = candle_nn::group_norm(${groups}, ${channels}, 1e-5, vb.pp("${block.id}"))?;`,
        },
        forward: `${outputVars[0]} = self.${block.id}.forward(&${inputVars[0]})?;`,
      };
    },
  },
};

export default GroupNorm;
