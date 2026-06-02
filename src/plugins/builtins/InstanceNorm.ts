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
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const dims = inputShape.length;
      let initExpr: string;
      if (dims <= 2) {
        const num = inputShape[inputShape.length - 1] ?? 0;
        initExpr = `nn.InstanceNorm1d(${num})`;
      } else {
        const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
        initExpr = `nn.InstanceNorm2d(${num})`;
      }
      return {
        attr: { name: block.id, init: initExpr },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      // Keras doesn't have a direct InstanceNorm; use GroupNormalization with groups=C
      const inputShape = block.inputShapes[0] ?? [];
      const channels = inputShape[inputShape.length - 3] ?? inputShape[1] ?? inputShape[inputShape.length - 1] ?? 0;
      return {
        attr: null,
        forward: `keras.layers.GroupNormalization(groups=${channels})(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const features =
        inputShape.length >= 4
          ? (inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0)
          : (inputShape[inputShape.length - 1] ?? 0);
      return {
        attr: {
          name: block.id,
          init: "",
          typeAnnotation: "candle_nn::BatchNorm",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default InstanceNorm;
