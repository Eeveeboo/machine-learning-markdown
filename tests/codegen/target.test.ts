import { describe, it, expect, beforeEach } from "vitest";
import { getTarget, registerTarget } from "../../src/codegen/target.js";
import type { CodegenTarget } from "../../src/codegen/target.js";

describe("codegen target registry", () => {
  it("returns undefined for unknown target", () => {
    expect(getTarget("pytorch")).toBeUndefined();
  });

  it("returns target after registration", () => {
    const target: CodegenTarget = {
      name: "pytorch",
      fileExtension: ".py",
      generate: () => [],
    };
    registerTarget(target);
    expect(getTarget("pytorch")).toBe(target);
  });

  it("overwrites existing target with same name", () => {
    const t1: CodegenTarget = { name: "onnx", fileExtension: ".onnx", generate: () => [] };
    const t2: CodegenTarget = { name: "onnx", fileExtension: ".onnx", generate: () => [{ path: "out.onnx", content: "data" }] };
    registerTarget(t1);
    registerTarget(t2);
    expect(getTarget("onnx")).toBe(t2);
  });
});
