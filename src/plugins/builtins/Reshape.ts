import type { BlockPlugin } from "../types.js";
import { getNumList } from "./_helpers.js";

const Reshape: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
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
    pytorch(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const shapeParam = block.params["shape"];
      if (shapeParam && shapeParam.kind === "shape") {
        const dims = shapeParam.dims.join(", ");
        return {
          attr: null,
          forward: `${x}.reshape(${x}.size(0), ${dims})`,
        };
      }
      return { attr: null, forward: `${x}.reshape(${x}.size(0), -1)` };
    },
    keras(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const shapeParam = block.params["shape"];
      if (shapeParam && shapeParam.kind === "shape") {
        const dims = shapeParam.dims.join(", ");
        return { attr: null, forward: `keras.layers.Reshape((${dims},))(${x})` };
      }
      return { attr: null, forward: `keras.layers.Reshape((-1,))(${x})` };
    },
    candle(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const dims = getNumList(block.params, "shape");
      const shape = dims.length ? dims.join(", ") : "0";
      return { attr: null, forward: `${x}.reshape(&[${shape}])?` };
    },
  },
};

export default Reshape;
