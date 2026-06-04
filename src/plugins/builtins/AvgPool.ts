import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

function poolOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

/** @codegen candle: placeholder — AvgPool2d not directly supported in candle_nn::ops */
export const AvgPool: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "kernel", type: "number", required: true }, { name: "stride", type: "number", required: false }, { name: "padding", type: "number", required: false }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("AvgPool requires an input");
    const [C, H, W] = inputs[0];
    const K = getNum(params, "kernel") ?? NaN;
    const S = getNum(params, "stride") ?? 1;
    const P = getNum(params, "padding") ?? 0;
    return [[C, poolOut(H, K, S, P), poolOut(W, K, S, P)]];
  },

  paramCount() {
    return 0;
  },

  codegen: {
    pytorch(block, inputVars, outputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      return {
        init: `self.${block.id} = nn.AvgPool2d(${kernel}, stride=${stride})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(block, inputVars, outputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      return {
        forward: `${outputVars[0]} = keras.layers.AveragePooling2D(pool_size=${kernel}, strides=${stride})(${inputVars[0]})`,
      };
    },

    candle(_block, inputVars, outputVars) {
      // Candle does not have a built-in avg_pool2d; emit a comment placeholder
      return {
        forward: `${outputVars[0]} = /* AvgPool2d not directly supported in candle_nn::ops */ ${inputVars[0]}.clone()?`,
      };
    },
  },
};

export default AvgPool;
