import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

export const GRU: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output", "hidden"],

  params: [{ name: "hidden_size", type: "number", required: true }, { name: "num_layers", type: "number", required: false }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("GRU requires an input");
    const [seq] = inputs[0];
    const H = getNum(params, "hidden_size") ?? NaN;
    return [[seq, H], [H]];
  },

  paramCount(inputs, params) {
    const inSize = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
    const H = getNum(params, "hidden_size") ?? NaN;
    const L = getNum(params, "num_layers") ?? 1;
    const firstLayer = 3 * (inSize * H + H * H + 2 * H);
    const extraLayer = Math.max(0, L - 1) * 3 * (H * H + H * H + 2 * H);
    return firstLayer + extraLayer;
  },

  codegen: {
    pytorch(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden =
        (block.params["hidden_size"]?.kind === "number" ? block.params["hidden_size"].value : undefined) ??
        (block.params["hidden"]?.kind === "number" ? block.params["hidden"].value : undefined) ??
        0;
      const layers =
        (block.params["num_layers"]?.kind === "number" ? block.params["num_layers"].value : undefined) ?? 1;
      return {
        init: `self.${block.id} = nn.GRU(${inputSize}, ${hidden}, num_layers=${layers}, batch_first=True)`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})[0]`,
      };
    },

    keras(block, inputVars, outputVars) {
      const hidden =
        (block.params["hidden_size"]?.kind === "number" ? block.params["hidden_size"].value : undefined) ??
        (block.params["hidden"]?.kind === "number" ? block.params["hidden"].value : undefined) ??
        0;
      return {
        forward: `${outputVars[0]} = keras.layers.GRU(${hidden}, return_sequences=True)(${inputVars[0]})`,
      };
    },

    candle(block, inputVars, outputVars) {
      const inputShape = block.inputShapes[0] ?? [];
      const inputSize = inputShape[inputShape.length - 1] ?? 0;
      const hidden =
        (block.params["hidden_size"]?.kind === "number" ? block.params["hidden_size"].value : undefined) ??
        (block.params["hidden"]?.kind === "number" ? block.params["hidden"].value : undefined) ??
        0;
      return {
        init: {
          field: `${block.id}: candle_nn::GRU`,
          body: `let ${block.id} = candle_nn::gru(${inputSize}, ${hidden}, candle_nn::GRUConfig::default(), vb.pp("${block.id}"))?;`,
        },
        forward: `let ${outputVars[0]} = { let states = candle_nn::RNN::seq(&self.${block.id}, &${inputVars[0]})?; candle_nn::RNN::states_to_tensor(&self.${block.id}, &states)? };`,
      };
    },
  },
};

export default GRU;
