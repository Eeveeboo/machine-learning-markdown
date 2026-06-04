import type { BlockPlugin } from "../types.js";
import { getNumList } from "./_helpers.js";

const Pad: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "padding", type: "shape", required: true }],
  inferShape(inputs, params) {
    if (inputs.length === 0) throw new Error("Pad requires an input");
    const [C, H, W] = inputs[0];
    const v = params["padding"];
    let p: number[] | null = null;
    if (v && v.kind === "shape") p = v.dims;
    else if (v && v.kind === "list") {
      p = v.items.map((i) => {
        if (i.kind === "number") return i.value;
        throw new Error("Expected number in list for padding");
      });
    }
    if (!p || p.length < 4) throw new Error("Pad requires padding=(P0,P1,P2,P3)");
    return [[C, H + p[0] + p[1], W + p[2] + p[3]]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const padding = block.params["padding"];
      if (padding && padding.kind === "list") {
        const vals = padding.items
          .filter((i) => i.kind === "number")
          .map((i) => (i as { kind: "number"; value: number }).value);
        return { forward: `${outputVars[0]} = torch.nn.functional.pad(${inputVars[0]}, (${vals.join(", ")}))` };
      }
      return { forward: `${outputVars[0]} = torch.nn.functional.pad(${inputVars[0]}, (0, 0))` };
    },
    keras(block, inputVars, outputVars) {
      const p = getNumList(block.params, "padding");
      if (p.length >= 4) {
        return {
          forward: `${outputVars[0]} = keras.layers.ZeroPadding2D(padding=((${p[0]}, ${p[1]}), (${p[2]}, ${p[3]})))(${inputVars[0]})`,
        };
      }
      return { forward: `${outputVars[0]} = keras.layers.ZeroPadding2D()(${inputVars[0]})` };
    },
    candle(block, inputVars, outputVars) {
      const p = getNumList(block.params, "padding");
      if (p.length >= 4) {
        return {
          forward: `${outputVars[0]} = ${inputVars[0]}.pad_with_zeros(2, ${p[0]}, ${p[1]})?.pad_with_zeros(3, ${p[2]}, ${p[3]})?`,
        };
      }
      return { forward: `${outputVars[0]} = ${inputVars[0]}.pad_with_zeros(2, 0, 0)?;` };
    },
  },
};

export default Pad;
