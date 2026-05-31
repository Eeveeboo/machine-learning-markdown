import type { Shape } from "../ast/graph.js";
import type { ParamValue } from "../ast/nodes.js";
import { registerBlock } from "./registry.js";

function num(params: Record<string, ParamValue>, key: string, def: number): number {
  const v = params[key];
  if (v == null) return def;
  if (v.kind === "number") return v.value;
  throw new Error(`Expected number for param ${key}`);
}

function shape(params: Record<string, ParamValue>, key: string): number[] {
  const v = params[key];
  if (v == null) throw new Error(`Missing required shape param ${key}`);
  if (v.kind === "shape") return v.dims;
  throw new Error(`Expected shape for param ${key}`);
}

function convOut(size: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((size + 2 * padding - kernel) / stride) + 1;
}

export function registerLayers(): void {
  // Input
  registerBlock({
    name: "Input",
    params: [{ name: "shape", type: "shape", required: true }],
    inferShape(_inputs, params) {
      return [shape(params, "shape")];
    },
    paramCount: () => 0,
  });

  // Output
  registerBlock({
    name: "Output",
    params: [],
    inferShape(inputs, _params) {
      if (inputs.length === 0) throw new Error("Output requires an input");
      return [inputs[0]];
    },
    paramCount: () => 0,
  });

  // Linear
  registerBlock({
    name: "Linear",
    params: [{ name: "out_features", type: "number", required: true }],
    inferShape(inputs, params) {
      if (inputs.length === 0) throw new Error("Linear requires an input");
      const outFeatures = num(params, "out_features", NaN);
      const inp = inputs[0];
      return [[...inp.slice(0, -1), outFeatures]];
    },
    paramCount(inputs, params) {
      const inF = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][inputs[0].length - 1] : 0;
      const outF = num(params, "out_features", NaN);
      return inF * outF + outF;
    },
  });

  // Conv1d
  registerBlock({
    name: "Conv1d",
    params: [
      { name: "filters", type: "number", required: true },
      { name: "kernel", type: "number", required: true },
      { name: "stride", type: "number", required: false, default: { kind: "number", value: 1, loc: { line: 0, col: 0, offset: 0 } } },
      { name: "padding", type: "number", required: false, default: { kind: "number", value: 0, loc: { line: 0, col: 0, offset: 0 } } },
    ],
    inferShape(inputs, params) {
      if (inputs.length === 0) throw new Error("Conv1d requires an input");
      const [, L] = inputs[0];
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      const S = num(params, "stride", 1);
      const P = num(params, "padding", 0);
      return [[F, convOut(L, K, S, P)]];
    },
    paramCount(inputs, params) {
      const inC = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0;
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      return inC * F * K + F;
    },
  });

  // Conv2d
  registerBlock({
    name: "Conv2d",
    params: [
      { name: "filters", type: "number", required: true },
      { name: "kernel", type: "number", required: true },
      { name: "stride", type: "number", required: false, default: { kind: "number", value: 1, loc: { line: 0, col: 0, offset: 0 } } },
      { name: "padding", type: "number", required: false, default: { kind: "number", value: 0, loc: { line: 0, col: 0, offset: 0 } } },
    ],
    inferShape(inputs, params) {
      if (inputs.length === 0) throw new Error("Conv2d requires an input");
      const [, H, W] = inputs[0];
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      const S = num(params, "stride", 1);
      const P = num(params, "padding", 0);
      return [[F, convOut(H, K, S, P), convOut(W, K, S, P)]];
    },
    paramCount(inputs, params) {
      const inC = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0;
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      return inC * F * K * K + F;
    },
  });

  // Conv3d
  registerBlock({
    name: "Conv3d",
    params: [
      { name: "filters", type: "number", required: true },
      { name: "kernel", type: "number", required: true },
      { name: "stride", type: "number", required: false, default: { kind: "number", value: 1, loc: { line: 0, col: 0, offset: 0 } } },
      { name: "padding", type: "number", required: false, default: { kind: "number", value: 0, loc: { line: 0, col: 0, offset: 0 } } },
    ],
    inferShape(inputs, params) {
      if (inputs.length === 0) throw new Error("Conv3d requires an input");
      const [, D, H, W] = inputs[0];
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      const S = num(params, "stride", 1);
      const P = num(params, "padding", 0);
      return [[F, convOut(D, K, S, P), convOut(H, K, S, P), convOut(W, K, S, P)]];
    },
    paramCount(inputs, params) {
      const inC = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0;
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      return inC * F * K * K * K + F;
    },
  });

  // TransposedConv2d
  registerBlock({
    name: "TransposedConv2d",
    params: [
      { name: "filters", type: "number", required: true },
      { name: "kernel", type: "number", required: true },
      { name: "stride", type: "number", required: false, default: { kind: "number", value: 1, loc: { line: 0, col: 0, offset: 0 } } },
      { name: "padding", type: "number", required: false, default: { kind: "number", value: 0, loc: { line: 0, col: 0, offset: 0 } } },
    ],
    inferShape(inputs, params) {
      if (inputs.length === 0) throw new Error("TransposedConv2d requires an input");
      const [, H, W] = inputs[0];
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      const S = num(params, "stride", 1);
      const P = num(params, "padding", 0);
      return [[F, (H - 1) * S - 2 * P + K, (W - 1) * S - 2 * P + K]];
    },
    paramCount(inputs, params) {
      const inC = inputs.length > 0 && inputs[0].length > 0 ? inputs[0][0] : 0;
      const F = num(params, "filters", NaN);
      const K = num(params, "kernel", NaN);
      return inC * F * K * K + F;
    },
  });

  // Embedding
  registerBlock({
    name: "Embedding",
    params: [
      { name: "vocab_size", type: "number", required: true },
      { name: "embed_dim", type: "number", required: true },
    ],
    inferShape(inputs, params) {
      if (inputs.length === 0) throw new Error("Embedding requires an input");
      const [seqLen] = inputs[0];
      const D = num(params, "embed_dim", NaN);
      return [[seqLen, D]];
    },
    paramCount(_inputs, params) {
      const vocab = num(params, "vocab_size", NaN);
      const dim = num(params, "embed_dim", NaN);
      return vocab * dim;
    },
  });
}
