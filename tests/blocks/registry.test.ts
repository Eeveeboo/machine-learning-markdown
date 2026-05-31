import { describe, it, expect, beforeEach } from "vitest";
import { registry, registerBlock, lookupBlock } from "../../src/blocks/index.js";
import type { BlockDef } from "../../src/blocks/index.js";

const dummyDef: BlockDef = {
  name: "linear",
  params: [
    { name: "out_features", type: "number", required: true },
    { name: "bias", type: "bool", required: false, default: { kind: "bool", value: true, loc: { line: 0, col: 0, offset: 0 } } },
  ],
  inferShape([input], _params) {
    return [[input[0], _params["out_features"] as unknown as number]];
  },
};

describe("block registry", () => {
  beforeEach(() => {
    registry.clear();
  });

  it("registers and looks up a block", () => {
    registerBlock(dummyDef);
    const found = lookupBlock("linear");
    expect(found).toBe(dummyDef);
  });

  it("returns undefined for unknown block", () => {
    expect(lookupBlock("unknown")).toBeUndefined();
  });

  it("silently overwrites duplicate registration", () => {
    registerBlock(dummyDef);
    const updated: BlockDef = { ...dummyDef, params: [] };
    registerBlock(updated);
    expect(lookupBlock("linear")).toBe(updated);
    expect(lookupBlock("linear")?.params).toHaveLength(0);
  });

  it("can register multiple distinct blocks", () => {
    const conv: BlockDef = {
      name: "conv2d",
      params: [],
      inferShape: () => [],
    };
    registerBlock(dummyDef);
    registerBlock(conv);
    expect(lookupBlock("linear")).toBe(dummyDef);
    expect(lookupBlock("conv2d")).toBe(conv);
  });
});
