import type { BlockPlugin } from "../types.js";
import { getNumList } from "./_helpers.js";

export const AdaptiveAvgPool: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("AdaptiveAvgPool requires an input");
    const [C] = inputs[0];
    const size = getNumList(params, "size");
    if (!size || size.length < 2) throw new Error("AdaptiveAvgPool requires size=(H,W)");
    return [[C, size[0], size[1]]];
  },

  paramCount() {
    return 0;
  },

  codegen: {
    pytorch(block, inputVars) {
      const sizeParam = block.params["size"];
      let outH = 1;
      let outW = 1;
      if (sizeParam?.kind === "shape" && sizeParam.dims.length >= 2) {
        outH = sizeParam.dims[0];
        outW = sizeParam.dims[1];
      } else if (sizeParam?.kind === "list") {
        const nums = sizeParam.items.filter((i) => i.kind === "number") as { kind: "number"; value: number }[];
        if (nums.length >= 2) {
          outH = nums[0].value;
          outW = nums[1].value;
        }
      } else {
        outH = (block.params["output_size_h"]?.kind === "number" ? block.params["output_size_h"].value : undefined) ??
               (block.params["output_size"]?.kind === "number" ? block.params["output_size"].value : undefined) ??
               1;
        outW = (block.params["output_size_w"]?.kind === "number" ? block.params["output_size_w"].value : undefined) ??
               outH;
      }
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: { name: block.id, init: `nn.AdaptiveAvgPool2d((${outH}, ${outW}))` },
        forward: `self.${block.id}(${mainIn})`,
      };
    },

    keras(block, inputVars) {
      // Keras doesn't have AdaptiveAvgPool2D; use Lambda or AveragePooling2D approximation
      const sizeParam = block.params["size"];
      let outH = 1;
      let outW = 1;
      if (sizeParam?.kind === "shape" && sizeParam.dims.length >= 2) {
        outH = sizeParam.dims[0];
        outW = sizeParam.dims[1];
      }
      const mainIn = inputVars[0] ?? "x";
      return {
        attr: null,
        forward: `keras.layers.Lambda(lambda x: tf.image.resize(x, (${outH}, ${outW})))(${mainIn})`,
      };
    },

    candle(_block, inputVars) {
      const mainIn = inputVars[0] ?? "x";
      // Candle doesn't have adaptive_avg_pool2d; use mean over spatial dims
      return {
        attr: null,
        forward: `${mainIn}.mean_keepdim(candle_core::D::Minus1)?.mean_keepdim(candle_core::D::Minus2)?`,
      };
    },
  },
};

export default AdaptiveAvgPool;
