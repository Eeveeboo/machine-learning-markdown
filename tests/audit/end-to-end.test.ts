import { describe, it, expect } from "vitest";
import { tokenize, parse, buildGraph, inferShapes, getTarget, registry } from "../../src/index.js";
import "../../src/blocks/index.js"; // registers all built-in blocks
import "../../src/codegen/targets/pytorch.js";
import "../../src/codegen/targets/keras.js";
import "../../src/codegen/targets/candle.js";

function pipeline(src: string) {
  const tokens = tokenize(src);
  const { nodes, errors } = parse(tokens);
  expect(errors).toHaveLength(0);
  const graph = buildGraph(nodes);
  const result = inferShapes(graph, registry);
  expect(result.errors).toHaveLength(0);
  return result.graph;
}

describe("end-to-end: three-target codegen", () => {
  // A simple Linear model
  const source = `
Input(shape=(10,))
Linear(out_features=64)
ReLU
Linear(out_features=32)
ReLU
Output
`.trim();

  it("pytorch target generates valid Python", () => {
    const graph = pipeline(source);
    const target = getTarget("pytorch");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map(f => f.content).join("\n");
    expect(code).toContain("import torch");
    expect(code).toContain("nn.Module");
  });

  it("keras target generates valid Python", () => {
    const graph = pipeline(source);
    const target = getTarget("keras");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map(f => f.content).join("\n");
    expect(code).toContain("keras.Input");
  });

  it("candle target generates valid Rust", () => {
    const graph = pipeline(source);
    const target = getTarget("candle");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map(f => f.content).join("\n");
    expect(code).toContain("use candle_core");
    expect(code).toContain("candle_nn::Linear");
  });
});
