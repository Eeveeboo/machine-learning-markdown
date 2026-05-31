import { describe, it, expect, beforeEach } from "vitest";
import "../../src/codegen/targets/candle.js";
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

describe("candle codegen target", () => {
  let target: ReturnType<typeof getTarget>;

  beforeEach(() => {
    target = getTarget("candle");
  });

  it("is registered", () => {
    expect(target).toBeDefined();
    expect(target!.name).toBe("candle");
    expect(target!.fileExtension).toBe(".rs");
  });

  describe("LeNet-style model", () => {
    it("generates valid Rust with use candle_core, struct, new, and forward", () => {
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
      expect(path).toBe("LeNet.rs");

      // imports
      expect(content).toContain("use candle_core");
      expect(content).toContain("use candle_nn");

      // struct
      expect(content).toContain("pub struct LeNet");

      // new fn
      expect(content).toContain("pub fn new(vb: VarBuilder) -> Result<Self>");
      expect(content).toContain(`candle_nn::conv2d(1, 6, 5`);
      expect(content).toContain(`candle_nn::conv2d(6, 16, 5`);
      expect(content).toContain(`candle_nn::linear(256, 120`);
      expect(content).toContain(`candle_nn::linear(120, 84`);
      expect(content).toContain(`candle_nn::linear(84, 10`);

      // forward fn
      expect(content).toContain("pub fn forward(&self, x: &Tensor) -> Result<Tensor>");
      expect(content).toContain(".relu()");
      expect(content).toContain("flatten_from(1)");
      expect(content).toContain("Ok(x)");
    });
  });

  describe("block mappings", () => {
    it("maps Dropout correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64]]),
        makeBlock("b1", "Dropout", { p: num(0.3) }, [[1, 64]], [[1, 64]]),
        makeBlock("b2", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("candle_nn::Dropout::new(0.3)");
    });

    it("maps Embedding correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10]]),
        makeBlock("b1", "Embedding", { vocab_size: num(1000), embed_dim: num(128) }, [[1, 10]], [[1, 10, 128]]),
        makeBlock("b2", "Output", {}, [[1, 10, 128]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("candle_nn::embedding(1000, 128");
    });

    it("maps BatchNorm correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64, 28, 28]]),
        makeBlock("b1", "BatchNorm", {}, [[1, 64, 28, 28]], [[1, 64, 28, 28]]),
        makeBlock("b2", "Output", {}, [[1, 64, 28, 28]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("candle_nn::batch_norm(64");
    });

    it("maps LSTM correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10, 32]]),
        makeBlock("b1", "LSTM", { hidden_size: num(64) }, [[1, 10, 32]], [[1, 10, 64]]),
        makeBlock("b2", "Output", {}, [[1, 10, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("candle_nn::lstm(32, 64");
    });

    it("maps Add to tensor addition", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64]]),
        makeBlock("b1", "Input", {}, [], [[1, 64]]),
        makeBlock("b2", "Add", {}, [[1, 64], [1, 64]], [[1, 64]]),
        makeBlock("b3", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [
        { from: "b0", to: "b2", tensorName: "a" },
        { from: "b1", to: "b2", tensorName: "b" },
        { from: "b2", to: "b3" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("&a + &b");
    });

    it("uses class name from graph groups", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], []), makeBlock("b1", "Output", {}, [], [])];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const graph = makeGraph(blocks, edges, [{ path: ["MyNet"], blockIds: ["b0", "b1"] }]);
      const files = target!.generate(graph, new Map());
      expect(files[0].path).toBe("MyNet.rs");
      expect(files[0].content).toContain("pub struct MyNet");
    });

    it("defaults to Model when no groups", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], []), makeBlock("b1", "Output", {}, [], [])];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const files = target!.generate(makeGraph(blocks, edges), new Map());
      expect(files[0].content).toContain("pub struct Model");
    });
  });
});
