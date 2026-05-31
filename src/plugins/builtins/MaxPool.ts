import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

function poolOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

export const MaxPool: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("MaxPool requires an input");
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
    pytorch(block, inputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: { name: block.id, init: `nn.MaxPool2d(${kernel}, stride=${stride})` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.MaxPooling2D(pool_size=${kernel}, strides=${stride})(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `candle_nn::ops::max_pool2d(&${mainIn}, ${kernel}, ${stride})?`,
      };
    },
  },
};

export default MaxPool;
