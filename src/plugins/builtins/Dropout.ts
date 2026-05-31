import type { BlockPlugin } from "../types.js";

const Dropout: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Dropout requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars) {
      const p = (() => {
        const v = block.params["p"] ?? block.params["rate"];
        return v && v.kind === "number" ? v.value : 0.5;
      })();
      return {
        attr: { name: block.id, init: `nn.Dropout(p=${p})` },
        forward: `self.${block.id}(${inputVars[0] ?? "x"})`,
      };
    },
    keras(_block, inputVars) {
      const x = inputVars[0] ?? "x";
      const p = (() => {
        const v = _block.params["p"] ?? _block.params["rate"];
        return v && v.kind === "number" ? v.value : 0.5;
      })();
      return {
        attr: null,
        forward: `keras.layers.Dropout(${p})(${x})`,
      };
    },
    candle(block, inputVars) {
      const x = inputVars[0] ?? "x";
      const p = (() => {
        const v = block.params["p"] ?? block.params["rate"];
        return v && v.kind === "number" ? v.value : 0.5;
      })();
      return {
        attr: { name: block.id, init: `candle_nn::Dropout::new(${p})`, typeAnnotation: "candle_nn::Dropout" },
        forward: `self.${block.id}.forward(&${x}, true)?`,
      };
    },
  },
};

export default Dropout;
