import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

export const Linear: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "out_features", type: "number", required: true }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("Linear requires an input");
    const outFeatures = getNum(params, "out_features") ?? NaN;
    const inp = inputs[0];
    return [[...inp.slice(0, -1), outFeatures]];
  },

  paramCount(inputs, params) {
    const inF = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
    const outF = getNum(params, "out_features") ?? NaN;
    return inF * outF + outF;
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const outputShape = block.outputShapes[0] ?? [];
      const inF = inputShape[inputShape.length - 1] ?? 0;
      const outF =
        (block.params["out_features"]?.kind === "number" ? block.params["out_features"].value : undefined) ??
        (block.params["units"]?.kind === "number" ? block.params["units"].value : undefined) ??
        outputShape[outputShape.length - 1] ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: { name: block.id, init: `nn.Linear(${inF}, ${outF})` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const outF =
        (block.params["out_features"]?.kind === "number" ? block.params["out_features"].value : undefined) ??
        (block.params["units"]?.kind === "number" ? block.params["units"].value : undefined) ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.Dense(${outF})(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const outputShape = block.outputShapes[0] ?? [];
      const inF = inputShape[inputShape.length - 1] ?? 0;
      const outF =
        (block.params["out_features"]?.kind === "number" ? block.params["out_features"].value : undefined) ??
        (block.params["units"]?.kind === "number" ? block.params["units"].value : undefined) ??
        outputShape[outputShape.length - 1] ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: {
          name: block.id,
          init: "",
          typeAnnotation: "candle_nn::Linear",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default Linear;
