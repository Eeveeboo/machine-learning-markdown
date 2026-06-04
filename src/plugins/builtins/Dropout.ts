import type { BlockPlugin } from "../types.js";

const Dropout: BlockPlugin = {
  inputs: ["x"],
  showDepth: false,
  outputs: ["y"],

  params: [{ name: "p", type: "number", required: false }],
  inferShape(inputs, _params) {
    if (inputs.length === 0) throw new Error("Dropout requires an input");
    return [inputs[0]];
  },
  paramCount: () => 0,
  codegen: {
    pytorch(block, inputVars, outputVars) {
      const p = (() => {
        const v = block.params["p"] ?? block.params["rate"];
        return v && v.kind === "number" ? v.value : 0.5;
      })();
      return {
        init: `self.${block.id} = nn.Dropout(p=${p})`,
        forward: `${outputVars[0]} = self.${block.id}(${inputVars[0]})`,
      };
    },
    keras(_block, inputVars, outputVars) {
      const p = (() => {
        const v = _block.params["p"] ?? _block.params["rate"];
        return v && v.kind === "number" ? v.value : 0.5;
      })();
      return {
        forward: `${outputVars[0]} = keras.layers.Dropout(${p})(${inputVars[0]})`,
      };
    },
    candle(block, inputVars, outputVars) {
      const p = (() => {
        const v = block.params["p"] ?? block.params["rate"];
        return v && v.kind === "number" ? v.value : 0.5;
      })();
      return {
        init: {
          field: `${block.id}: candle_nn::Dropout`,
          body: `let ${block.id} = candle_nn::Dropout::new(${p});`,
        },
        forward: `${outputVars[0]} = self.${block.id}.forward(&${inputVars[0]}, true)?`,
      };
    },
  },
};

export default Dropout;
