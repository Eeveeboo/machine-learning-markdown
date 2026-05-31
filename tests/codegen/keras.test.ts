import { describe, it, expect, beforeEach } from "vitest";
import "../../src/codegen/targets/keras.js";
import { getTarget } from "../../src/codegen/target.js";
import type { Graph, Block, Edge } from "../../src/ast/graph.js";

const loc = { line: 1, col: 1, offset: 0 };

function makeBlock(id: string, type: string, params: Block["params"] = {}, inputShapes: number[][] = [], outputShapes: number[][] = []): Block {
  return { id, type, params, inputShapes, outputShapes, loc };
}

function num(value: number) {
  return { kind: "number" as const, value, loc };
}

function makeGraph(blocks: Block[], edges: Edge[] = [], groups: Graph["groups"] = []): Graph {
  return { blocks, edges, groups };
}

describe("keras codegen target", () => {
  let target: ReturnType<typeof getTarget>;

  beforeEach(() => {
    target = getTarget("keras");
  });

  it("is registered", () => {
    expect(target).toBeDefined();
    expect(target!.name).toBe("keras");
    expect(target!.fileExtension).toBe(".py");
  });

  describe("LeNet-style model", () => {
    it("generates valid Keras code with keras.Input, keras.Model, and build_model()", () => {
      const blocks: Block[] = [
        makeBlock("block_0", "Input", {}, [], [[1, 1, 28, 28]]),
        makeBlock("block_1", "Conv2d", { filters: num(6), kernel: num(5) }, [[1, 1, 28, 28]], [[1, 6, 24, 24]]),
        makeBlock("block_2", "ReLU", {}, [[1, 6, 24, 24]], [[1, 6, 24, 24]]),
        makeBlock("block_3", "MaxPool", { kernel: num(2), stride: num(2) }, [[1, 6, 24, 24]], [[1, 6, 12, 12]]),
        makeBlock("block_4", "Conv2d", { filters: num(16), kernel: num(5) }, [[1, 6, 12, 12]], [[1, 16, 8, 8]]),
        makeBlock("block_5", "ReLU", {}, [[1, 16, 8, 8]], [[1, 16, 8, 8]]),
        makeBlock("block_6", "MaxPool", { kernel: num(2), stride: num(2) }, [[1, 16, 8, 8]], [[1, 16, 4, 4]]),
        makeBlock("block_7", "Flatten", {}, [[1, 16, 4, 4]], [[1, 256]]),
        makeBlock("block_8", "Linear", { out_features: num(120) }, [[1, 256]], [[1, 120]]),
        makeBlock("block_9", "ReLU", {}, [[1, 120]], [[1, 120]]),
        makeBlock("block_10", "Linear", { out_features: num(84) }, [[1, 120]], [[1, 84]]),
        makeBlock("block_11", "ReLU", {}, [[1, 84]], [[1, 84]]),
        makeBlock("block_12", "Linear", { out_features: num(10) }, [[1, 84]], [[1, 10]]),
        makeBlock("block_13", "Output", {}, [[1, 10]], []),
      ];

      const edges: Edge[] = [
        { from: "block_0", to: "block_1" },
        { from: "block_1", to: "block_2" },
        { from: "block_2", to: "block_3" },
        { from: "block_3", to: "block_4" },
        { from: "block_4", to: "block_5" },
        { from: "block_5", to: "block_6" },
        { from: "block_6", to: "block_7" },
        { from: "block_7", to: "block_8" },
        { from: "block_8", to: "block_9" },
        { from: "block_9", to: "block_10" },
        { from: "block_10", to: "block_11" },
        { from: "block_11", to: "block_12" },
        { from: "block_12", to: "block_13" },
      ];

      const graph = makeGraph(blocks, edges, [{ path: ["LeNet"], blockIds: blocks.map((b) => b.id) }]);
      const files = target!.generate(graph, new Map());

      expect(files).toHaveLength(1);
      const { path, content } = files[0];
      expect(path).toBe("LeNet.py");

      // imports
      expect(content).toContain("import tensorflow as tf");
      expect(content).toContain("from tensorflow import keras");

      // function
      expect(content).toContain("def build_model():");

      // keras.Input
      expect(content).toContain("keras.Input(shape=");

      // keras.Model
      expect(content).toContain("keras.Model(inputs=");

      // layers
      expect(content).toContain("keras.layers.Conv2D(6");
      expect(content).toContain("keras.layers.Conv2D(16");
      expect(content).toContain("keras.layers.MaxPooling2D");
      expect(content).toContain("keras.layers.Flatten()");
      expect(content).toContain("keras.layers.Dense(120)");
      expect(content).toContain("keras.layers.Dense(84)");
      expect(content).toContain("keras.layers.Dense(10)");
      expect(content).toContain("keras.layers.ReLU()");
    });
  });

  describe("ResNet-style branching graph", () => {
    it("generates named tensor variables for branches", () => {
      const blocks: Block[] = [
        makeBlock("block_0", "Input", {}, [], [[1, 56, 56, 64]]),
        makeBlock("block_1", "Conv2d", { filters: num(64), kernel: num(3) }, [[1, 56, 56, 64]], [[1, 56, 56, 64]]),
        makeBlock("block_2", "Conv2d", { filters: num(64), kernel: num(3) }, [[1, 56, 56, 64]], [[1, 56, 56, 64]]),
        makeBlock("block_3", "Add", {}, [[1, 56, 56, 64], [1, 56, 56, 64]], [[1, 56, 56, 64]]),
        makeBlock("block_4", "Output", {}, [[1, 56, 56, 64]], []),
      ];

      const edges: Edge[] = [
        { from: "block_0", to: "block_1", tensorName: "identity" },
        { from: "block_0", to: "block_3" },
        { from: "block_1", to: "block_2" },
        { from: "block_2", to: "block_3" },
        { from: "block_3", to: "block_4" },
      ];

      const graph = makeGraph(blocks, edges);
      const files = target!.generate(graph, new Map());
      const { content } = files[0];

      expect(content).toContain("identity");
      expect(content).toContain("keras.layers.Add()([");
    });
  });

  describe("block mappings", () => {
    it("maps Linear correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 128]]),
        makeBlock("b1", "Linear", { out_features: num(64) }, [[1, 128]], [[1, 64]]),
        makeBlock("b2", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.Dense(64)");
    });

    it("maps Dropout correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64]]),
        makeBlock("b1", "Dropout", { p: num(0.5) }, [[1, 64]], [[1, 64]]),
        makeBlock("b2", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.Dropout(0.5)");
    });

    it("maps LSTM correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10, 32]]),
        makeBlock("b1", "LSTM", { hidden_size: num(64) }, [[1, 10, 32]], [[1, 10, 64]]),
        makeBlock("b2", "Output", {}, [[1, 10, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.LSTM(64");
    });

    it("maps Embedding correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10]]),
        makeBlock("b1", "Embedding", { vocab_size: num(1000), embed_dim: num(128) }, [[1, 10]], [[1, 10, 128]]),
        makeBlock("b2", "Output", {}, [[1, 10, 128]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.Embedding(1000, 128)");
    });

    it("maps BatchNorm correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64, 28, 28]]),
        makeBlock("b1", "BatchNorm", {}, [[1, 64, 28, 28]], [[1, 64, 28, 28]]),
        makeBlock("b2", "Output", {}, [[1, 64, 28, 28]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.BatchNormalization()");
    });

    it("maps LayerNorm correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 128]]),
        makeBlock("b1", "LayerNorm", {}, [[1, 128]], [[1, 128]]),
        makeBlock("b2", "Output", {}, [[1, 128]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.LayerNormalization()");
    });

    it("maps Concat to keras.layers.Concatenate", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 32]]),
        makeBlock("b1", "Input", {}, [], [[1, 32]]),
        makeBlock("b2", "Concat", {}, [[1, 32], [1, 32]], [[1, 64]]),
        makeBlock("b3", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [
        { from: "b0", to: "b2", tensorName: "a" },
        { from: "b1", to: "b2", tensorName: "b" },
        { from: "b2", to: "b3" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("keras.layers.Concatenate(");
    });

    it("uses model name from graph groups for filename", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], []), makeBlock("b1", "Output", {}, [], [])];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const graph = makeGraph(blocks, edges, [{ path: ["MyNet"], blockIds: ["b0", "b1"] }]);
      const files = target!.generate(graph, new Map());
      expect(files[0].path).toBe("MyNet.py");
    });

    it("defaults to Model filename when no groups", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], []), makeBlock("b1", "Output", {}, [], [])];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const files = target!.generate(makeGraph(blocks, edges), new Map());
      expect(files[0].path).toBe("Model.py");
    });
  });
});
