import { describe, it, expect } from "vitest";
import { lookupBlock } from "../../src/blocks/index.js";
import type { ParamValue } from "../../src/ast/nodes.js";

function n(value: number): ParamValue {
  return { kind: "number", value, loc: { line: 0, col: 0, offset: 0 } };
}

function s(...dims: number[]): ParamValue {
  return { kind: "shape", dims, loc: { line: 0, col: 0, offset: 0 } };
}

function infer(name: string, inputs: number[][], params: Record<string, ParamValue> = {}): number[][] {
  const def = lookupBlock(name);
  if (!def) throw new Error(`Block ${name} not found`);
  return def.inferShape(inputs, params);
}

describe("activation blocks", () => {
  const activations = ["ReLU", "SELU", "GELU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "PReLU"];
  for (const name of activations) {
    it(`${name} is passthrough`, () => {
      expect(infer(name, [[3, 32, 32]])).toEqual([[3, 32, 32]]);
    });
  }

  it("Softmax accepts axis param", () => {
    expect(infer("Softmax", [[10]], { axis: n(0) })).toEqual([[10]]);
  });
});

describe("norm blocks", () => {
  const norms = ["BatchNorm", "LayerNorm", "GroupNorm", "InstanceNorm"];
  for (const name of norms) {
    it(`${name} is passthrough`, () => {
      expect(infer(name, [[64, 8, 8]])).toEqual([[64, 8, 8]]);
    });
  }
});

describe("pool blocks", () => {
  it("MaxPool reduces spatial dims", () => {
    // (1, 28, 28) kernel=2 stride=2 → (1, 14, 14)
    expect(infer("MaxPool", [[1, 28, 28]], { kernel: n(2), stride: n(2) })).toEqual([[1, 14, 14]]);
  });

  it("MaxPool with padding", () => {
    // (1, 32, 32) kernel=3 stride=1 padding=1 → (1, 32, 32)
    expect(infer("MaxPool", [[1, 32, 32]], { kernel: n(3), stride: n(1), padding: n(1) })).toEqual([[1, 32, 32]]);
  });

  it("AvgPool same formula as MaxPool", () => {
    expect(infer("AvgPool", [[3, 10, 10]], { kernel: n(2), stride: n(2) })).toEqual([[3, 5, 5]]);
  });

  it("GlobalAvgPool → (C,1,1)", () => {
    expect(infer("GlobalAvgPool", [[64, 7, 7]])).toEqual([[64, 1, 1]]);
  });

  it("AdaptiveAvgPool → (C,H,W) from size param", () => {
    expect(infer("AdaptiveAvgPool", [[512, 7, 7]], { size: s(4, 4) })).toEqual([[512, 4, 4]]);
  });
});

describe("recurrent blocks", () => {
  it("LSTM returns two output shapes", () => {
    const out = infer("LSTM", [[20, 128]], { hidden_size: n(256) });
    expect(out).toEqual([[20, 256], [256]]);
  });

  it("GRU returns two output shapes", () => {
    const out = infer("GRU", [[10, 64]], { hidden_size: n(128) });
    expect(out).toEqual([[10, 128], [128]]);
  });

  it("RNN returns single output shape", () => {
    const out = infer("RNN", [[15, 32]], { hidden_size: n(64) });
    expect(out).toEqual([[15, 64]]);
  });
});

describe("transform blocks", () => {
  it("Flatten: (C,H,W) → (C*H*W,)", () => {
    expect(infer("Flatten", [[6, 5, 5]])).toEqual([[150]]);
  });

  it("Flatten: (D,) stays (D,)", () => {
    expect(infer("Flatten", [[100]])).toEqual([[100]]);
  });

  it("Reshape returns shape param", () => {
    expect(infer("Reshape", [[120]], { shape: s(4, 30) })).toEqual([[4, 30]]);
  });

  it("Dropout is passthrough", () => {
    expect(infer("Dropout", [[256]], { rate: n(0.5) })).toEqual([[256]]);
  });

  it("Pad adds padding to H and W", () => {
    // (1,4,4) + padding (1,1,1,1) → (1,6,6)
    expect(infer("Pad", [[1, 4, 4]], { padding: s(1, 1, 1, 1) })).toEqual([[1, 6, 6]]);
  });
});

describe("merge blocks", () => {
  it("Add returns input shape", () => {
    expect(infer("Add", [[3, 32, 32], [3, 32, 32]])).toEqual([[3, 32, 32]]);
  });

  it("Mul returns input shape", () => {
    expect(infer("Mul", [[10], [10]])).toEqual([[10]]);
  });

  it("Sub returns input shape", () => {
    expect(infer("Sub", [[5, 5], [5, 5]])).toEqual([[5, 5]]);
  });

  it("Div returns input shape", () => {
    expect(infer("Div", [[8], [8]])).toEqual([[8]]);
  });

  it("Concat axis=0 sums channel dim", () => {
    expect(infer("Concat", [[64, 28, 28], [32, 28, 28]], { axis: n(0) })).toEqual([[96, 28, 28]]);
  });

  it("Concat default axis=1", () => {
    // axis=1 (default) concatenates second dim: [16,10,10]+[16,10,10] → [16,20,10]
    expect(infer("Concat", [[16, 10, 10], [16, 10, 10]])).toEqual([[16, 20, 10]]);
  });

  it("Concat axis=0", () => {
    expect(infer("Concat", [[10, 5], [20, 5]], { axis: n(0) })).toEqual([[30, 5]]);
  });

  it("MatMul (A,B) x (B,C) → (A,C)", () => {
    expect(infer("MatMul", [[4, 8], [8, 16]])).toEqual([[4, 16]]);
  });
});

describe("extra/control flow blocks", () => {
  it("Split divides axis dim by chunks", () => {
    const out = infer("Split", [[12, 4]], { chunks: n(3), axis: n(0) });
    expect(out).toEqual([[4, 4], [4, 4], [4, 4]]);
  });

  it("Split defaults to axis=0", () => {
    const out = infer("Split", [[6, 8]], { chunks: n(2) });
    expect(out).toEqual([[3, 8], [3, 8]]);
  });

  it("Repeat is passthrough", () => {
    expect(infer("Repeat", [[5, 5]], { times: n(3) })).toEqual([[5, 5]]);
  });

  it("Map is passthrough", () => {
    expect(infer("Map", [[128]])).toEqual([[128]]);
  });

  it("Gather is passthrough", () => {
    expect(infer("Gather", [[1000]], { axis: n(0) })).toEqual([[1000]]);
  });
});

describe("LeNet end-to-end shape trace", () => {
  it("traces shapes from Input → Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → Flatten → Linear → Output", () => {
    // Input: (1, 28, 28)
    let shape = infer("Input", [], { shape: s(1, 28, 28) })[0];
    expect(shape).toEqual([1, 28, 28]);

    // Conv2d filters=6 kernel=5 → (6, 24, 24)
    shape = infer("Conv2d", [shape], { filters: n(6), kernel: n(5) })[0];
    expect(shape).toEqual([6, 24, 24]);

    // ReLU passthrough
    shape = infer("ReLU", [shape])[0];
    expect(shape).toEqual([6, 24, 24]);

    // MaxPool kernel=2 stride=2 → (6, 12, 12)
    shape = infer("MaxPool", [shape], { kernel: n(2), stride: n(2) })[0];
    expect(shape).toEqual([6, 12, 12]);

    // Conv2d filters=16 kernel=5 → (16, 8, 8)
    shape = infer("Conv2d", [shape], { filters: n(16), kernel: n(5) })[0];
    expect(shape).toEqual([16, 8, 8]);

    // ReLU
    shape = infer("ReLU", [shape])[0];
    expect(shape).toEqual([16, 8, 8]);

    // MaxPool kernel=2 stride=2 → (16, 4, 4)
    shape = infer("MaxPool", [shape], { kernel: n(2), stride: n(2) })[0];
    expect(shape).toEqual([16, 4, 4]);

    // Flatten → (256,)
    shape = infer("Flatten", [shape])[0];
    expect(shape).toEqual([256]);

    // Linear 120 → (120,)
    shape = infer("Linear", [shape], { out_features: n(120) })[0];
    expect(shape).toEqual([120]);

    // Linear 84 → (84,)
    shape = infer("Linear", [shape], { out_features: n(84) })[0];
    expect(shape).toEqual([84]);

    // Linear 10 → (10,)
    shape = infer("Linear", [shape], { out_features: n(10) })[0];
    expect(shape).toEqual([10]);

    // Output
    shape = infer("Output", [shape])[0];
    expect(shape).toEqual([10]);
  });
});
