import type { BlockPlugin } from "../types.js";
import { getNumList } from "./_helpers.js";

const Reshape: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "shape", type: "shape", required: true }],
  inferShape(_inputs, params) {
    const v = params["shape"];
    if (!v) throw new Error("Reshape requires shape param");
    if (v.kind === "shape") return [v.dims];
    if (v.kind === "list") {
      const dims = v.items.map((i) => {
        if (i.kind === "number") return i.value;
        throw new Error("Expected number in list for shape");
      });
      return [dims];
    }
    throw new Error("Reshape requires shape param");
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const shapeParam = block.params["shape"];
      if (shapeParam && shapeParam.kind === "shape") {
        const dims = shapeParam.dims.join(", ");
        return {
          forward: `${outputVars[0]} = ${inputVars[0]}.reshape(${inputVars[0]}.size(0), ${dims})`,
        };
      }
      return { forward: `${outputVars[0]} = ${inputVars[0]}.reshape(${inputVars[0]}.size(0), -1)` };
    },
    keras(block, inputVars, outputVars) {
      const shapeParam = block.params["shape"];
      if (shapeParam && shapeParam.kind === "shape") {
        const dims = shapeParam.dims.join(", ");
        return { forward: `${outputVars[0]} = keras.layers.Reshape((${dims},))(${inputVars[0]})` };
      }
      return { forward: `${outputVars[0]} = keras.layers.Reshape((-1,))(${inputVars[0]})` };
    },
    candle(block, inputVars, outputVars) {
      const dims = getNumList(block.params, "shape");
      const shape = dims.length ? dims.join(", ") : "0";
      return { forward: `${outputVars[0]} = ${inputVars[0]}.reshape(&[${shape}])?;` };
    },
  },
};

export default Reshape;
