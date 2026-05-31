import { describe, it, expect, beforeEach } from "vitest";
import { registry, lookupBlock } from "../../src/blocks/index.js";
import { registerLayers } from "../../src/blocks/layers.js";
import type { ParamValue } from "../../src/ast/nodes.js";

function n(value: number): ParamValue {
  return { kind: "number", value, loc: { line: 0, col: 0, offset: 0 } };
}

function s(...dims: number[]): ParamValue {
  return { kind: "shape", dims, loc: { line: 0, col: 0, offset: 0 } };
}

function infer(name: string, inputs: number[][], params: Record<string, ParamValue>): number[][] {
  const def = lookupBlock(name);
  if (!def) throw new Error(`Block ${name} not found`);
  return def.inferShape(inputs, params);
}

describe("layer blocks", () => {
  beforeEach(() => {
    registry.clear();
    registerLayers();
  });

  describe("Input", () => {
    it("returns shape param as output", () => {
      expect(infer("Input", [], { shape: s(1, 28, 28) })).toEqual([[1, 28, 28]]);
    });
    it("works with 1D shape", () => {
      expect(infer("Input", [], { shape: s(512) })).toEqual([[512]]);
    });
  });

  describe("Output", () => {
    it("passes through input shape", () => {
      expect(infer("Output", [[3, 32, 32]], {})).toEqual([[3, 32, 32]]);
    });
    it("throws without input", () => {
      expect(() => infer("Output", [], {})).toThrow();
    });
  });

  describe("Linear", () => {
    it("replaces last dim", () => {
      expect(infer("Linear", [[128, 256]], { out_features: n(10) })).toEqual([[128, 10]]);
    });
    it("works with 1D input", () => {
      expect(infer("Linear", [[256]], { out_features: n(10) })).toEqual([[10]]);
    });
    it("works with 3D input", () => {
      expect(infer("Linear", [[32, 128, 256]], { out_features: n(64) })).toEqual([[32, 128, 64]]);
    });
  });

  describe("Conv1d", () => {
    it("basic conv", () => {
      expect(infer("Conv1d", [[1, 100]], { filters: n(16), kernel: n(3) })).toEqual([[16, 98]]);
    });
    it("with padding", () => {
      expect(infer("Conv1d", [[1, 100]], { filters: n(16), kernel: n(3), padding: n(1) })).toEqual([[16, 100]]);
    });
    it("with stride=2", () => {
      expect(infer("Conv1d", [[1, 100]], { filters: n(8), kernel: n(3), stride: n(2) })).toEqual([[8, 49]]);
    });
  });

  describe("Conv2d", () => {
    it("Conv2d(kernel=5, filters=6) on (1,28,28) → (6,24,24)", () => {
      expect(infer("Conv2d", [[1, 28, 28]], { filters: n(6), kernel: n(5) })).toEqual([[6, 24, 24]]);
    });
    it("with stride=2", () => {
      expect(infer("Conv2d", [[3, 32, 32]], { filters: n(16), kernel: n(3), stride: n(2) })).toEqual([[16, 15, 15]]);
    });
    it("with padding=1", () => {
      expect(infer("Conv2d", [[1, 28, 28]], { filters: n(6), kernel: n(3), padding: n(1) })).toEqual([[6, 28, 28]]);
    });
  });

  describe("Conv3d", () => {
    it("basic 3D conv", () => {
      expect(infer("Conv3d", [[1, 10, 10, 10]], { filters: n(4), kernel: n(3) })).toEqual([[4, 8, 8, 8]]);
    });
    it("with padding", () => {
      expect(infer("Conv3d", [[1, 8, 8, 8]], { filters: n(4), kernel: n(3), padding: n(1) })).toEqual([[4, 8, 8, 8]]);
    });
  });

  describe("TransposedConv2d", () => {
    it("basic transposed conv (stride=1 is identity of conv)", () => {
      // H'=(H-1)*1 - 0 + K = H-1+K
      expect(infer("TransposedConv2d", [[6, 24, 24]], { filters: n(1), kernel: n(5) })).toEqual([[1, 28, 28]]);
    });
    it("with stride=2", () => {
      // H'=(15-1)*2 - 0 + 3 = 28+3 = 31
      expect(infer("TransposedConv2d", [[16, 15, 15]], { filters: n(3), kernel: n(3), stride: n(2) })).toEqual([[3, 31, 31]]);
    });
  });

  describe("Embedding", () => {
    it("maps (seq_len,) → (seq_len, D)", () => {
      expect(infer("Embedding", [[50]], { vocab_size: n(10000), embed_dim: n(128) })).toEqual([[50, 128]]);
    });
  });
});
