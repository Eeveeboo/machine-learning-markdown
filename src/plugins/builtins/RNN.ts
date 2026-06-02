import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

/** @codegen candle: placeholder — vanilla RNN not natively supported in candle_nn; uses linear approximation */
export const RNN: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "hidden_size", type: "number", required: true }, { name: "num_layers", type: "number", required: false }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("RNN requires an input");
    const [seq] = inputs[0];
    const H = getNum(params, "hidden_size") ?? NaN;
    return [[seq, H]];
  },

  paramCount(inputs, params) {
    const inSize = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
    const H = getNum(params, "hidden_size") ?? NaN;
    const L = getNum(params, "num_layers") ?? 1;
    const firstLayer = 1 * (inSize * H + H * H + 2 * H);
    const extraLayer = Math.max(0, L - 1) * 1 * (H * H + H * H + 2 * H);
    return firstLayer + extraLayer;
  },

  codegen: {
    pytorch(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden =
        (block.params["hidden_size"]?.kind === "number" ? block.params["hidden_size"].value : undefined) ??
        (block.params["hidden"]?.kind === "number" ? block.params["hidden"].value : undefined) ??
        0;
      const layers =
        (block.params["num_layers"]?.kind === "number" ? block.params["num_layers"].value : undefined) ?? 1;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: {
          name: block.id,
          init: `nn.RNN(${inputSize}, ${hidden}, num_layers=${layers}, batch_first=True)`,
        },
        forward: `self.${block.id}(${mainIn})[0]`,
      };
    },

    keras(block, inputVars) {
      const hidden =
        (block.params["hidden_size"]?.kind === "number" ? block.params["hidden_size"].value : undefined) ??
        (block.params["hidden"]?.kind === "number" ? block.params["hidden"].value : undefined) ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.SimpleRNN(${hidden}, return_sequences=True)(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden =
        (block.params["hidden_size"]?.kind === "number" ? block.params["hidden_size"].value : undefined) ??
        (block.params["hidden"]?.kind === "number" ? block.params["hidden"].value : undefined) ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: {
          name: block.id,
          // candle does not have a built-in vanilla RNN; emit a placeholder
          init: "",
          typeAnnotation: "candle_nn::Linear",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default RNN;
