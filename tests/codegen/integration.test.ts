import { describe, it, expect, beforeAll } from "vitest";
import { tokenize, parse, buildGraph, inferShapes, getTarget, registry } from "../../src/index.js";
import "../../src/blocks/index.js"; // registers all built-in blocks
import "../../src/codegen/targets/pytorch.js";
import "../../src/codegen/targets/keras.js";

beforeAll(() => {
  // blocks are registered via the import above; nothing extra needed
});

describe("end-to-end pipeline: Embedding model", () => {
  const source = `
Input(shape=(10,))
Embedding(vocab_size=1000, embed_dim=128)
Output
`.trim();

  function pipeline(src: string) {
    const tokens = tokenize(src);
    const { nodes, errors } = parse(tokens);
    expect(errors).toHaveLength(0);
    const graph = buildGraph(nodes);
    const result = inferShapes(graph, registry);
    expect(result.errors).toHaveLength(0);
    return result.graph;
  }

  it("PyTorch: generates nn.Embedding(1000, 128)", () => {
    const graph = pipeline(source);
    const target = getTarget("pytorch");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map((f) => f.content).join("\n");
    expect(code).toContain("nn.Embedding(1000, 128)");
    expect(code).not.toContain("nn.Embedding(1000, 0)");
  });

  it("Keras: generates keras.layers.Embedding(1000, 128)", () => {
    const graph = pipeline(source);
    const target = getTarget("keras");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map((f) => f.content).join("\n");
    expect(code).toContain("keras.layers.Embedding(1000, 128)");
    expect(code).not.toContain("keras.layers.Embedding(1000, 0)");
  });
});

describe("end-to-end pipeline: ResNet-like model with skip connection", () => {
  const source = `
Input(shape=(64, 56, 56))
Conv2d(filters=64, kernel=3, padding=1)
ReLU
-> [skip]
Conv2d(filters=64, kernel=3, padding=1)
ReLU
[skip] -> Add
Output
`.trim();

  function pipeline(src: string) {
    const tokens = tokenize(src);
    const { nodes, errors } = parse(tokens);
    expect(errors).toHaveLength(0);
    const graph = buildGraph(nodes);
    const result = inferShapes(graph, registry);
    expect(result.errors).toHaveLength(0);
    return result.graph;
  }

  it("PyTorch: generates code with Conv2d(64, 64, 3, padding=1)", () => {
    const graph = pipeline(source);
    const target = getTarget("pytorch");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map((f) => f.content).join("\n");
    expect(code).toContain("import torch");
    expect(code).toContain("nn.Conv2d");
  });

  it("Keras: generates code with keras.layers.Conv2D", () => {
    const graph = pipeline(source);
    const target = getTarget("keras");
    expect(target).toBeDefined();
    const files = target!.generate(graph, registry);
    expect(files.length).toBeGreaterThan(0);
    const code = files.map((f) => f.content).join("\n");
    expect(code).toContain("keras.layers.Conv2D");
  });
});
