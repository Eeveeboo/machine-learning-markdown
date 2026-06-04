import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

function poolOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

export const MaxPool: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "kernel", type: "number", required: true }, { name: "stride", type: "number", required: false }, { name: "padding", type: "number", required: false }],
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
    pytorch(block, inputVars, outputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      return {
        init: `self.${block.id} = nn.MaxPool2d(${kernel}, stride=${stride})`,
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
        init: null,
        forward: `${outputVars[0]} = keras.layers.MaxPooling2D(pool_size=${kernel}, strides=${stride})(${inputVars[0]})`,
      };
    },

    candle(block, inputVars, outputVars) {
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        2;
      const stride =
        (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ??
        kernel;
      const padding =
        (block.params["padding"]?.kind === "number" ? block.params["padding"].value : undefined) ??
        (block.params["pad"]?.kind === "number" ? block.params["pad"].value : undefined) ??
        0;
      // Candle's max_pool2d doesn't support padding; pad the input manually via pad_with_zeros
      const padPrefix = padding > 0
        ? `${inputVars[0]}.pad_with_zeros(2, ${padding}, ${padding})?.pad_with_zeros(3, ${padding}, ${padding})?`
        : inputVars[0];
      const method = stride !== kernel ? "max_pool2d_with_stride" : "max_pool2d";
      const args = stride !== kernel ? `${kernel}, ${stride}` : `${kernel}`;
      return {
        init: null,
        forward: `${outputVars[0]} = ${padPrefix}.${method}(${args})?;`,
      };
    },
  },
};

export default MaxPool;
