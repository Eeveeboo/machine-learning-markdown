import type { BlockPlugin } from "../types.js";
import { getNum, convOut } from "./_helpers.js";

export const Conv1d: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [
    { name: "filters", type: "number", required: true },
    { name: "kernel", type: "number", required: true },
    { name: "stride", type: "number", required: false },
    { name: "padding", type: "number", required: false },
  ],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("Conv1d requires an input");
    const [, L] = inputs[0];
    const F = getNum(params, "filters") ?? NaN;
    const K = getNum(params, "kernel") ?? NaN;
    const S = getNum(params, "stride") ?? 1;
    const P = getNum(params, "padding") ?? 0;
    return [[F, convOut(L, K, S, P)]];
  },

  paramCount(inputs, params) {
    const inC = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0;
    const F = getNum(params, "filters") ?? NaN;
    const K = getNum(params, "kernel") ?? NaN;
    return inC * F * K + F;
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const inCh = inputShape[inputShape.length - 2] ?? inputShape[1] ?? 0;
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
        attr: { name: block.id, init: `nn.Conv1d(${inCh}, ${filters}, ${kernel}, stride=${stride}, padding=${padding})` },
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
        forward: `keras.layers.Conv1D(${filters}, ${kernel}, strides=${stride}, padding='${safeP}')(${mainIn})`,
      };
    },

    candle(block, _inputVars) {
      // candle does not have a built-in Conv1d; fallback forward expression
      const inputShape = block.inputShapes[0] ?? [];
      const inCh = inputShape[inputShape.length - 2] ?? inputShape[1] ?? 0;
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
      const mainIn = _inputVars[0] ?? "x";
      return {
        attr: {
          name: block.id,
          init: `candle_nn::conv1d(${inCh}, ${filters}, ${kernel}, candle_nn::Conv1dConfig { stride: ${stride}, padding: ${padding}, ..Default::default() }, vb.pp("${block.id}"))?`,
          typeAnnotation: "candle_nn::Conv1d",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default Conv1d;
