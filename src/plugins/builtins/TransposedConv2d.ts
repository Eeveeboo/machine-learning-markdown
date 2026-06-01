import type { BlockPlugin } from "../types.js";
import { getNum, convTransposeOutputSize } from "./_helpers.js";

export const TransposedConv2d: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [
    { name: "filters", type: "number", required: true },
    { name: "kernel", type: "number", required: true },
    { name: "stride", type: "number", required: false },
    { name: "padding", type: "number", required: false },
  ],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("TransposedConv2d requires an input");
    const [, H, W] = inputs[0];
    const F = getNum(params, "filters") ?? NaN;
    const K = getNum(params, "kernel") ?? NaN;
    const S = getNum(params, "stride") ?? 1;
    const P = getNum(params, "padding") ?? 0;
    return [[F, convTransposeOutputSize(H, K, P, S), convTransposeOutputSize(W, K, P, S)]];
  },

  paramCount(inputs, params) {
    const inC = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0;
    const F = getNum(params, "filters") ?? NaN;
    const K = getNum(params, "kernel") ?? NaN;
    return inC * F * K * K + F;
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const inCh = inputShape[inputShape.length - 3] ?? inputShape[1] ?? 0;
      const filters =
        (block.params["filters"]?.kind === "number" ? block.params["filters"].value : undefined) ??
        (block.params["out_channels"]?.kind === "number" ? block.params["out_channels"].value : undefined) ??
        0;
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        3;
      const stride = (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ?? 1;
      const padding = (block.params["padding"]?.kind === "number" ? block.params["padding"].value : undefined) ?? 0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: { name: block.id, init: `nn.ConvTranspose2d(${inCh}, ${filters}, ${kernel}, stride=${stride}, padding=${padding})` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const filters =
        (block.params["filters"]?.kind === "number" ? block.params["filters"].value : undefined) ??
        (block.params["out_channels"]?.kind === "number" ? block.params["out_channels"].value : undefined) ??
        0;
      const kernel =
        (block.params["kernel"]?.kind === "number" ? block.params["kernel"].value : undefined) ??
        (block.params["kernel_size"]?.kind === "number" ? block.params["kernel_size"].value : undefined) ??
        3;
      const stride = (block.params["stride"]?.kind === "number" ? block.params["stride"].value : undefined) ?? 1;
      const paddingStr =
        (block.params["padding"]?.kind === "string" || block.params["padding"]?.kind === "bareword"
          ? (block.params["padding"] as { value: string }).value
          : undefined) ?? "valid";
      const mainIn = inputVars[0] ?? "x";
      const safeP = paddingStr.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n').replace(/\r/g, '\\r');
      return {
        attr: null,
        forward: `keras.layers.Conv2DTranspose(${filters}, ${kernel}, strides=${stride}, padding='${safeP}')(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      // candle does not have ConvTranspose2d in standard candle_nn; emit placeholder
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `unimplemented!("TransposedConv2d not supported in candle")  // ${mainIn}`,
      };
    },
  },
};

export default TransposedConv2d;
