import type { BlockPlugin } from "../types.js";

export const InstanceNorm: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("InstanceNorm requires an input");
    return [inputs[0]];
  },

  paramCount(inputs, _params) {
    if (inputs.length === 0 || inputs[0].length === 0) return 0;
    return inputs[0][0] * 2; // C * 2
  },

  codegen: {
    pytorch(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const dims = inputShape.length;
      let initExpr: string;
      if (dims <= 2) {
        const num = inputShape[inputShape.length - 1] ?? 0;
        initExpr = `self.${block.id} = nn.InstanceNorm1d(${num})`;
      } else {
        const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
        initExpr = `self.${block.id} = nn.InstanceNorm2d(${num})`;
      }
      return {
        init: initExpr,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(block, inputVars, outputVars) {
      // Keras doesn't have a direct InstanceNorm; use GroupNormalization with groups=C
      const inputShape = block.inputShapes[0] ?? [];
      const channels = inputShape[inputShape.length - 3] ?? inputShape[1] ?? inputShape[inputShape.length - 1] ?? 0;
      return {
        init: null,
        forward: `${outputVars[0]} = keras.layers.GroupNormalization(groups=${channels})(${inputVars[0]})`,
      };
    },

    candle(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const features =
        inputShape.length >= 4
          ? (inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0)
          : (inputShape[inputShape.length - 1] ?? 0);
      return {
        init: {
          field: `${block.id}: candle_nn::BatchNorm`,
          body: `let ${block.id} = candle_nn::batch_norm(${features}, 1e-5, vb.pp("${block.id}"))?;`,
        },
        forward: `${outputVars[0]} = self.${block.id}.forward(&${inputVars[0]})?;`,
      };
    },
  },
};

export default InstanceNorm;
