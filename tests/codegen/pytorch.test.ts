import { describe, it, expect, beforeEach } from "vitest";
import "../../src/codegen/targets/pytorch.js";
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

describe("pytorch codegen target", () => {
  let target: ReturnType<typeof getTarget>;

  beforeEach(() => {
    target = getTarget("pytorch");
  });

  it("is registered", () => {
    expect(target).toBeDefined();
    expect(target!.name).toBe("pytorch");
    expect(target!.fileExtension).toBe(".py");
  });

  describe("LeNet-style model", () => {
    it("generates valid Python with imports and forward()", () => {
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
      expect(content).toContain("import torch");
      expect(content).toContain("import torch.nn as nn");

      // class definition
      expect(content).toContain("class LeNet(nn.Module):");
      expect(content).toContain("def __init__(self):");
      expect(content).toContain("super().__init__()");
      expect(content).toContain("def forward(self, x):");

      // layers
      expect(content).toContain("nn.Conv2d(1, 6, 5");
      expect(content).toContain("nn.Conv2d(6, 16, 5");
      expect(content).toContain("nn.MaxPool2d(2");
      expect(content).toContain("nn.Flatten()");
      expect(content).toContain("nn.Linear(256, 120)");
      expect(content).toContain("nn.Linear(120, 84)");
      expect(content).toContain("nn.Linear(84, 10)");

      // forward has return
      expect(content).toContain("return ");
    });
  });

  describe("ResNet-style branching graph", () => {
    it("generates named tensor variables for branches", () => {
      const blocks: Block[] = [
        makeBlock("block_0", "Input", {}, [], [[1, 64, 56, 56]]),
        makeBlock("block_1", "Conv2d", { filters: num(64), kernel: num(3) }, [[1, 64, 56, 56]], [[1, 64, 56, 56]]),
        makeBlock("block_2", "Conv2d", { filters: num(64), kernel: num(3) }, [[1, 64, 56, 56]], [[1, 64, 56, 56]]),
        makeBlock("block_3", "Add", {}, [[1, 64, 56, 56], [1, 64, 56, 56]], [[1, 64, 56, 56]]),
        makeBlock("block_4", "Output", {}, [[1, 64, 56, 56]], []),
      ];

      // block_0 → block_1 (main path), block_0 → skip (residual), block_1 → block_2, block_2 + skip → Add
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

      // Named tensor should appear
      expect(content).toContain("identity");
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
      expect(content).toContain("nn.Linear(128, 64)");
    });

    it("maps Dropout correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64]]),
        makeBlock("b1", "Dropout", { p: num(0.5) }, [[1, 64]], [[1, 64]]),
        makeBlock("b2", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("nn.Dropout(p=0.5)");
    });

    it("maps LSTM correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10, 32]]),
        makeBlock("b1", "LSTM", { hidden_size: num(64), num_layers: num(2) }, [[1, 10, 32]], [[1, 10, 64]]),
        makeBlock("b2", "Output", {}, [[1, 10, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("nn.LSTM(32, 64, num_layers=2");
      // LSTM output is [0] (output only, not hidden state)
      expect(content).toContain("self.b1(x)[0]");
    });

    it("maps Embedding correctly", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 10]]),
        makeBlock("b1", "Embedding", { vocab_size: num(1000), embed_dim: num(128) }, [[1, 10]], [[1, 10, 128]]),
        makeBlock("b2", "Output", {}, [[1, 10, 128]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("nn.Embedding(1000, 128)");
    });

    it("maps BatchNorm2d for 4D input", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64, 28, 28]]),
        makeBlock("b1", "BatchNorm", {}, [[1, 64, 28, 28]], [[1, 64, 28, 28]]),
        makeBlock("b2", "Output", {}, [[1, 64, 28, 28]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("nn.BatchNorm2d(64)");
    });

    it("maps BatchNorm1d for 2D input", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64]]),
        makeBlock("b1", "BatchNorm", {}, [[1, 64]], [[1, 64]]),
        makeBlock("b2", "Output", {}, [[1, 64]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("nn.BatchNorm1d(64)");
    });

    it("maps Concat to torch.cat", () => {
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
      expect(content).toContain("torch.cat(");
    });

    it("uses class name from graph groups", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], []), makeBlock("b1", "Output", {}, [], [])];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const graph = makeGraph(blocks, edges, [{ path: ["MyNet"], blockIds: ["b0", "b1"] }]);
      const files = target!.generate(graph, new Map());
      expect(files[0].path).toBe("MyNet.py");
      expect(files[0].content).toContain("class MyNet(nn.Module):");
    });

    it("defaults to Model class when no groups", () => {
      const blocks = [makeBlock("b0", "Input", {}, [], []), makeBlock("b1", "Output", {}, [], [])];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const graph = makeGraph(blocks, edges);
      const files = target!.generate(graph, new Map());
      expect(files[0].content).toContain("class Model(nn.Module):");
    });
  });

  describe("forward parameters (multi-input support)", () => {
    it("single unnamed input uses forward(self, x)", () => {
      const blocks = [makeBlock("b0", "Input"), makeBlock("b1", "Output")];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("def forward(self, x):");
    });

    it("single named input uses tensorName as param", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 64]]),
        makeBlock("b1", "Linear", { out_features: num(32) }, [[1, 64]], [[1, 32]]),
        makeBlock("b2", "Output", {}, [[1, 32]], []),
      ];
      const edges: Edge[] = [
        { from: "b0", to: "b1", tensorName: "features" },
        { from: "b1", to: "b2" },
      ];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("def forward(self, features):");
      expect(content).toContain("self.b1(features)");
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
      expect(content).toContain("def forward(self, query, key, value):");
      expect(content).toContain("self.b_linear_q(query)");
      expect(content).toContain("self.b_linear_k(key)");
      expect(content).toContain("self.b_linear_v(value)");
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
      expect(content).toContain("def forward(self, x, x2, x3):");
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
      expect(content).toContain("def forward(self, data, x, x2):");
    });

    it("emits shape comments for input params", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 3, 224, 224]]),
        makeBlock("b1", "Output"),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      expect(content).toContain("# x: input");
      expect(content).toContain("1, 3, 224, 224");
    });
  });

  describe("shape annotations", () => {
    it("includes shape comments in forward()", () => {
      const blocks = [
        makeBlock("b0", "Input", {}, [], [[1, 1, 28, 28]]),
        makeBlock("b1", "Conv2d", { filters: num(6), kernel: num(5) }, [[1, 1, 28, 28]], [[1, 6, 24, 24]]),
        makeBlock("b2", "Output", {}, [[1, 6, 24, 24]], []),
      ];
      const edges: Edge[] = [{ from: "b0", to: "b1" }, { from: "b1", to: "b2" }];
      const content = target!.generate(makeGraph(blocks, edges), new Map())[0].content;
      // Should have shape annotations
      expect(content).toContain("1, 6, 24, 24");
    });
  });
});
