import type { BlockPlugin } from "../types.js";

export const BatchNorm: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "num_features", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("BatchNorm requires an input");
    return [inputs[0]];
  },

  paramCount(inputs, params) {
    const nf =
      params["num_features"]?.kind === "number"
        ? params["num_features"].value
        : inputs.length > 0 && inputs[0].length > 0
          ? inputs[0][0]
          : 0;
    return nf * 2;
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const mainIn = inputVars[0] ?? "x";
      const dims = inputShape.length;
      let initExpr: string;
      if (dims <= 2) {
        const num = inputShape[inputShape.length - 1] ?? 0;
        initExpr = `nn.BatchNorm1d(${num})`;
      } else {
        const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
        initExpr = `nn.BatchNorm2d(${num})`;
      }
      return {
        attr: { name: block.id, init: initExpr },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.BatchNormalization()(${mainIn})`,
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

export default BatchNorm;
