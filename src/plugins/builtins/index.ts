import { registerBlock } from "../../blocks/registry.js";
import { registerBlockCodegen } from "../../codegen/block-codegen.js";
import type { BlockPlugin } from "../types.js";

import Input from "./Input.js";
import Output from "./Output.js";
import Linear from "./Linear.js";
import Conv1d from "./Conv1d.js";
import Conv2d from "./Conv2d.js";
import Conv3d from "./Conv3d.js";
import TransposedConv2d from "./TransposedConv2d.js";
import Embedding from "./Embedding.js";
import ReLU from "./ReLU.js";
import LeakyReLU from "./LeakyReLU.js";
import PReLU from "./PReLU.js";
import ELU from "./ELU.js";
import SELU from "./SELU.js";
import GELU from "./GELU.js";
import SiLU from "./SiLU.js";
import Sigmoid from "./Sigmoid.js";
import Tanh from "./Tanh.js";
import Softmax from "./Softmax.js";
import BatchNorm from "./BatchNorm.js";
import LayerNorm from "./LayerNorm.js";
import GroupNorm from "./GroupNorm.js";
import InstanceNorm from "./InstanceNorm.js";
import MaxPool from "./MaxPool.js";
import AvgPool from "./AvgPool.js";
import GlobalAvgPool from "./GlobalAvgPool.js";
import AdaptiveAvgPool from "./AdaptiveAvgPool.js";
import LSTM from "./LSTM.js";
import GRU from "./GRU.js";
import RNN from "./RNN.js";
import Dropout from "./Dropout.js";
import Flatten from "./Flatten.js";
import Reshape from "./Reshape.js";
import Pad from "./Pad.js";
import Add from "./Add.js";
import Mul from "./Mul.js";
import Sub from "./Sub.js";
import Div from "./Div.js";
import Concat from "./Concat.js";
import MatMul from "./MatMul.js";
import Split from "./Split.js";
import Repeat from "./Repeat.js";
import MapPlugin from "./Map.js";
import Gather from "./Gather.js";

const builtins: Record<string, BlockPlugin> = {
  Input,
  Output,
  Linear,
  Conv1d,
  Conv2d,
  Conv3d,
  TransposedConv2d,
  Embedding,
  ReLU,
  LeakyReLU,
  PReLU,
  ELU,
  SELU,
  GELU,
  SiLU,
  Sigmoid,
  Tanh,
  Softmax,
  BatchNorm,
  LayerNorm,
  GroupNorm,
  InstanceNorm,
  MaxPool,
  AvgPool,
  GlobalAvgPool,
  AdaptiveAvgPool,
  LSTM,
  GRU,
  RNN,
  Dropout,
  Flatten,
  Reshape,
  Pad,
  Add,
  Mul,
  Sub,
  Div,
  Concat,
  MatMul,
  Split,
  Repeat,
  Map: MapPlugin,
  Gather,
};

export function registerBuiltins(): void {
  for (const [name, plugin] of Object.entries(builtins)) {
    registerBlock({
      name,
      params: plugin.params ?? [],
      inferShape: plugin.inferShape ?? ((inputs) => inputs),
      paramCount: plugin.paramCount,
      showDepth: plugin.showDepth,
    });

    if (plugin.codegen) {
      for (const [target, fn] of Object.entries(plugin.codegen)) {
        registerBlockCodegen(name, target, fn);
      }
    }
  }
}

// Register on import as a side-effect
registerBuiltins();
