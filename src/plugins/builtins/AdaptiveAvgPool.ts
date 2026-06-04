import type { BlockPlugin } from "../types.js";
import { getNumList } from "./_helpers.js";

export const AdaptiveAvgPool: BlockPlugin = {
  inputs: ["input"],
  outputs: ["output"],

  params: [{ name: "size", type: "shape", required: true }],
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
    pytorch(block, inputVars, outputVars) {
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
      return {
        init: `self.${block.id} = nn.AdaptiveAvgPool2d((${outH}, ${outW}))`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },

    keras(block, inputVars, outputVars) {
      // Keras doesn't have AdaptiveAvgPool2D; use Lambda or AveragePooling2D approximation
      const sizeParam = block.params["size"];
      let outH = 1;
      let outW = 1;
      if (sizeParam?.kind === "shape" && sizeParam.dims.length >= 2) {
        outH = sizeParam.dims[0];
        outW = sizeParam.dims[1];
      }
      return {
        forward: `${outputVars[0]} = keras.layers.Lambda(lambda x: tf.image.resize(x, (${outH}, ${outW})))(${inputVars[0]})`,
      };
    },

    candle(_block, inputVars, outputVars) {
      // Candle doesn't have adaptive_avg_pool2d; use mean over spatial dims
      return {
        forward: `${outputVars[0]} = ${inputVars[0]}.mean_keepdim(candle_core::D::Minus1)?.mean_keepdim(candle_core::D::Minus2)?;`,
      };
    },
  },
};

export default AdaptiveAvgPool;
