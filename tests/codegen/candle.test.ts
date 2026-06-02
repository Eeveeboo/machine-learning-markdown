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

      // new fn — convenience, auto-generates scope names
      expect(content).toContain("pub fn new(weights: VarBuilder) -> Result<Self>");
      expect(content).toContain('Self::with_scopes(weights,');
      expect(content).toContain('"block_1",');
      expect(content).toContain('"block_4",');
      expect(content).toContain('"block_8",');
      expect(content).toContain('"block_10",');
      expect(content).toContain('"block_12",');
      // with_scopes — custom &str scope params
      expect(content).toContain("pub fn with_scopes(");
      expect(content).toContain("weights: VarBuilder,");
      expect(content).toContain("block_1: &str");
      expect(content).toContain("block_4: &str");
      expect(content).toContain("block_8: &str");
      expect(content).toContain("block_10: &str");
      expect(content).toContain("block_12: &str");
      // Init uses weights.pp(<param>) not weights.pp("<string>")
      expect(content).toContain("weights.pp(block_1)");

      // forward fn — typed input param
      expect(content).toContain("pub fn forward(&self,");
      expect(content).toContain("x: &Tensor");
      expect(content).toContain(") -> Result<Tensor>");
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

    it("LSTM scope uses snake_case param", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10, 32]]),
        makeBlock("b1", "LSTM", { hidden_size: num(64) }, [[1, 10, 32]], [[1, 10, 64]]),
        makeBlock("b2", "Output", {}, [[1, 10, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      // scope param is snake_case: b1 not B1
      expect(content).toContain("weights.pp(b1)");
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

  describe("typed forward parameters", () => {
    it("single unnamed input uses x: &Tensor", () => {
      const blocks = [makeBlock("b0", "Input"), makeBlock("b1", "Output")];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("x: &Tensor");
      const forwardMatch = content.match(/pub fn forward\(&self,([^)]+)\)/);
      expect(forwardMatch).not.toBeNull();
      const params = forwardMatch![1].trim();
      expect(params.split(",")).toHaveLength(1);
    });

    it("single named input uses tensorName", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], [[1, 64]]), makeBlock("b1", "Linear", { out_features: num(32) }, [[1, 64]], [[1, 32]]), makeBlock("b2", "Output", {}, [[1, 32]], [])];
      const edges: Edge[] = [
        { from: "b0", to: "b1", tensorName: "features" },
        { from: "b1", to: "b2" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("features: &Tensor");
      expect(content).toContain("self.b1.forward(&features)?");
    });

    it("multiple named inputs (Attention-style)", () => {
      const blocks = [
        makeBlock("b_q", "Input"),
        makeBlock("b_k", "Input"),
        makeBlock("b_v", "Input"),
        makeBlock("b_linear_q", "Linear", { out_features: num(64) }, [[1, 64]], [[1, 64]]),
        makeBlock("b_linear_k", "Linear", { out_features: num(64) }, [[1, 64]], [[1, 64]]),
        makeBlock("b_linear_v", "Linear", { out_features: num(64) }, [[1, 64]], [[1, 64]]),
        makeBlock("b_add", "Add", {}, [[1, 64], [1, 64]], [[1, 64]]),
        makeBlock("b_out", "Output"),
      ];
      const edges: Edge[] = [
        { from: "b_q", to: "b_linear_q", tensorName: "query" },
        { from: "b_k", to: "b_linear_k", tensorName: "key" },
        { from: "b_v", to: "b_linear_v", tensorName: "value" },
        { from: "b_linear_q", to: "b_add" },
        { from: "b_linear_k", to: "b_add" },
        { from: "b_linear_v", to: "b_add" },
        { from: "b_add", to: "b_out" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("query: &Tensor");
      expect(content).toContain("key: &Tensor");
      expect(content).toContain("value: &Tensor");
      expect(content).toContain("self.b_linear_q.forward(&query)?");
      expect(content).toContain("self.b_linear_k.forward(&key)?");
      expect(content).toContain("self.b_linear_v.forward(&value)?");
    });

    it("multiple unnamed inputs use x, x2, x3", () => {
      const blocks = [
        makeBlock("b0", "Input"),
        makeBlock("b1", "Input"),
        makeBlock("b2", "Input"),
        makeBlock("b3", "Output"),
      ];
      const edges: Edge[] = [
        { from: "b0", to: "b3" },
        { from: "b1", to: "b3" },
        { from: "b2", to: "b3" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("x: &Tensor");
      expect(content).toContain("x2: &Tensor");
      expect(content).toContain("x3: &Tensor");
    });

    it("mixed named and unnamed inputs", () => {
      const blocks = [
        makeBlock("b0", "Input"),
        makeBlock("b1", "Input"),
        makeBlock("b2", "Input"),
        makeBlock("b3", "Output"),
      ];
      const edges: Edge[] = [
        { from: "b0", to: "b3", tensorName: "data" },
        { from: "b1", to: "b3" },
        { from: "b2", to: "b3" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("data: &Tensor");
      expect(content).toContain("x: &Tensor");
      expect(content).toContain("x2: &Tensor");
    });
  });

  describe("known candle codegen gaps", () => {
    it("Conv3d emits unimplemented! for candle", () => {
      const blocks = [makeBlock("c1", "Conv3d", { filters: num(8), kernel: num(3) }, [[1, 3, 16, 16, 16]], [[1, 8, 14, 14, 14]])];
      const graph = makeGraph(blocks);
      const files = target!.generate(graph, new Map());
      const code = files.map(f => f.content).join("\n");
      expect(code).toContain("unimplemented!");
    });

    it("TransposedConv2d emits unimplemented! for candle", () => {
      const blocks = [makeBlock("tc1", "TransposedConv2d", { filters: num(8), kernel: num(3) }, [[1, 3, 16, 16]], [[1, 8, 18, 18]])];
      const graph = makeGraph(blocks);
      const files = target!.generate(graph, new Map());
      const code = files.map(f => f.content).join("\n");
      expect(code).toContain("unimplemented!");
    });

    it("PReLU falls back to .relu() for candle", () => {
      const blocks = [makeBlock("p1", "PReLU", {}, [[1, 64]], [[1, 64]])];
      const graph = makeGraph(blocks);
      const files = target!.generate(graph, new Map());
      const code = files.map(f => f.content).join("\n");
      expect(code).toContain(".relu()");
    });

    it("AvgPool emits comment about not directly supported for candle", () => {
      const blocks = [makeBlock("a1", "AvgPool", { kernel: num(2) }, [[1, 3, 28, 28]], [[1, 3, 14, 14]])];
      const graph = makeGraph(blocks);
      const files = target!.generate(graph, new Map());
      const code = files.map(f => f.content).join("\n");
      expect(code).toContain("not directly supported");
    });

    it("RNN emits comment about not natively supported and uses candle_nn::linear", () => {
      const blocks = [makeBlock("r1", "RNN", { hidden_size: num(64) }, [[1, 10, 32]], [[1, 10, 64]])];
      const graph = makeGraph(blocks);
      const files = target!.generate(graph, new Map());
      const code = files.map(f => f.content).join("\n");
      expect(code).toContain("not natively supported");
      expect(code).toContain("candle_nn::linear");
    });
  });
});
