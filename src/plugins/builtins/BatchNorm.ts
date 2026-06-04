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
    pytorch(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const dims = inputShape.length;
      let initExpr: string;
      if (dims <= 2) {
        const num = inputShape[inputShape.length - 1] ?? 0;
        initExpr = `self.${block.id} = nn.BatchNorm1d(${num})`;
      } else {
        const num = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
        initExpr = `self.${block.id} = nn.BatchNorm2d(${num})`;
      }
      return {
        init: initExpr,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(block, inputVars, outputVars) {
      return {
        forward: `${outputVars[0]} = keras.layers.BatchNormalization()(${inputVars[0]})`,
      };
    },

    candle(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      // MLMD shapes may include batch dim (4D: [N,C,H,W]) or not (3D: [C,H,W]).
      // Channels/features come from the second dim when batch is present, first dim otherwise.
      const features =
        inputShape.length >= 4
          ? (inputShape[1] ?? 0)  // [N, C, H, W] with batch
          : inputShape.length >= 3
            ? (inputShape[0] ?? 0)  // [C, H, W] without batch
            : (inputShape[inputShape.length - 1] ?? 0);  // [F] or [T, F]
      return {
        init: {
          field: `${block.id}: candle_nn::BatchNorm`,
          body: `let ${block.id} = candle_nn::batch_norm(${features}, 1e-5, vb.pp("${block.id}"))?;`,
        },
        forward: `${outputVars[0]} = self.${block.id}.forward_t(&${inputVars[0]}, false)?;`,
      };
    },
  },
};

export default BatchNorm;
