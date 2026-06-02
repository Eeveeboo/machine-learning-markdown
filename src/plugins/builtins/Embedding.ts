import type { BlockPlugin } from "../types.js";
import { getNum } from "./_helpers.js";

export const Embedding: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [
    { name: "vocab_size", type: "number", required: true },
    { name: "embed_dim", type: "number", required: true },
  ],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("Embedding requires an input");
    const [seqLen] = inputs[0];
    const D = getNum(params, "embed_dim") ?? NaN;
    return [[seqLen, D]];
  },

  paramCount(_inputs, params) {
    const vocab = getNum(params, "vocab_size") ?? NaN;
    const dim = getNum(params, "embed_dim") ?? NaN;
    return vocab * dim;
  },

  codegen: {
    pytorch(block, inputVars) {
      const vocab =
        (block.params["vocab_size"]?.kind === "number" ? block.params["vocab_size"].value : undefined) ?? 0;
      const embed =
        (block.params["embed_dim"]?.kind === "number" ? block.params["embed_dim"].value : undefined) ??
        (block.params["embedding_dim"]?.kind === "number" ? block.params["embedding_dim"].value : undefined) ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: { name: block.id, init: `nn.Embedding(${vocab}, ${embed})` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      const vocab =
        (block.params["vocab_size"]?.kind === "number" ? block.params["vocab_size"].value : undefined) ?? 0;
      const embed =
        (block.params["embed_dim"]?.kind === "number" ? block.params["embed_dim"].value : undefined) ??
        (block.params["embedding_dim"]?.kind === "number" ? block.params["embedding_dim"].value : undefined) ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.Embedding(${vocab}, ${embed})(${mainIn})`,
      };
    },

    candle(block, inputVars) {
      const vocab =
        (block.params["vocab_size"]?.kind === "number" ? block.params["vocab_size"].value : undefined) ?? 0;
      const dim =
        (block.params["embed_dim"]?.kind === "number" ? block.params["embed_dim"].value : undefined) ??
        (block.params["embedding_dim"]?.kind === "number" ? block.params["embedding_dim"].value : undefined) ??
        0;
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: {
          name: block.id,
          init: "",
          typeAnnotation: "candle_nn::Embedding",
        },
        forward: `self.${block.id}.forward(&${mainIn})?`,
      };
    },
  },
};

export default Embedding;
