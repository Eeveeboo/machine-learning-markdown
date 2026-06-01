import { describe, it, expect } from "vitest";
import { registry } from "../../src/blocks/registry.js";
// Side-effect: registers all built-in blocks
import "../../src/blocks/index.js";

const EXPECTED_BLOCKS: string[] = [
  "Input",
  "Output",
  "Linear",
  "Conv1d",
  "Conv2d",
  "Conv3d",
  "TransposedConv2d",
  "Embedding",
  "ReLU",
  "LeakyReLU",
  "PReLU",
  "ELU",
  "SELU",
  "GELU",
  "SiLU",
  "Sigmoid",
  "Tanh",
  "Softmax",
  "BatchNorm",
  "LayerNorm",
  "GroupNorm",
  "InstanceNorm",
  "MaxPool",
  "AvgPool",
  "GlobalAvgPool",
  "AdaptiveAvgPool",
  "LSTM",
  "GRU",
  "RNN",
  "Dropout",
  "Flatten",
  "Reshape",
  "Pad",
  "Add",
  "Mul",
  "Sub",
  "Div",
  "Concat",
  "MatMul",
  "Split",
  "Repeat",
  "Map",
  "Gather",
];

describe("built-in registry count", () => {
  it("should have exactly 43 registered built-in blocks", () => {
    const names = [...registry.keys()].sort();
    console.log(`\nRegistered blocks (${registry.size} total):`);
    for (const name of names) {
      console.log(`  - ${name}`);
    }

    expect(registry.size).toBe(43);

    // Verify every expected block name is present
    for (const expected of EXPECTED_BLOCKS) {
      expect(registry.has(expected), `Missing block: ${expected}`).toBe(true);
    }

    // Verify no unexpected blocks (beyond the expected list)
    const expectedSet = new Set(EXPECTED_BLOCKS);
    for (const name of names) {
      expect(
        expectedSet.has(name),
        `Unexpected block in registry: ${name}`,
      ).toBe(true);
    }
  });
});
