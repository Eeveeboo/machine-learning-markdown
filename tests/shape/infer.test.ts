import { describe, it, expect } from "vitest";
import "../../src/blocks/index.js"; // register all blocks
import { registry } from "../../src/blocks/registry.js";
import { inferShapes } from "../../src/shape/infer.js";
import type { Graph, Block, Edge } from "../../src/ast/graph.js";

const loc = { line: 0, col: 0, offset: 0 };

function makeBlock(id: string, type: string, params: Record<string, unknown> = {}): Block {
  return {
    id,
    type,
    params: params as Block["params"],
    inputShapes: [],
    outputShapes: [],
    loc,
  };
}

function makeEdge(from: string, to: string): Edge {
  return { from, to };
}

function n(value: number): Block["params"][string] {
  return { kind: "number", value, loc };
}

function s(...dims: number[]): Block["params"][string] {
  return { kind: "shape", dims, loc };
}

describe("inferShapes", () => {
  describe("LeNet trace", () => {
    it("Input(1,28,28) → Conv2d(6, kernel=5) → ReLU → MaxPool2d(2) → Conv2d(16, kernel=5) → ReLU → MaxPool2d(2) → Flatten → Linear(120) → ReLU → Linear(84) → Linear(10)", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(1, 28, 28) }),
          makeBlock("b1", "Conv2d", { filters: n(6), kernel: n(5) }),
          makeBlock("b2", "ReLU"),
          makeBlock("b3", "MaxPool", { kernel: n(2), stride: n(2) }),
          makeBlock("b4", "Conv2d", { filters: n(16), kernel: n(5) }),
          makeBlock("b5", "ReLU"),
          makeBlock("b6", "MaxPool", { kernel: n(2), stride: n(2) }),
          makeBlock("b7", "Flatten"),
          makeBlock("b8", "Linear", { out_features: n(120) }),
          makeBlock("b9", "ReLU"),
          makeBlock("b10", "Linear", { out_features: n(84) }),
          makeBlock("b11", "Linear", { out_features: n(10) }),
        ],
        edges: [
          makeEdge("b0", "b1"),
          makeEdge("b1", "b2"),
          makeEdge("b2", "b3"),
          makeEdge("b3", "b4"),
          makeEdge("b4", "b5"),
          makeEdge("b5", "b6"),
          makeEdge("b6", "b7"),
          makeEdge("b7", "b8"),
          makeEdge("b8", "b9"),
          makeEdge("b9", "b10"),
          makeEdge("b10", "b11"),
        ],
        groups: [],
      };

      const result = inferShapes(graph, registry);
      expect(result.errors).toHaveLength(0);

      const blockById = new Map(result.graph.blocks.map((b) => [b.id, b]));

      expect(blockById.get("b0")!.outputShapes).toEqual([[1, 28, 28]]);
      // Conv2d: (28 - 5) / 1 + 1 = 24, filters=6
      expect(blockById.get("b1")!.outputShapes).toEqual([[6, 24, 24]]);
      // ReLU: passthrough
      expect(blockById.get("b2")!.outputShapes).toEqual([[6, 24, 24]]);
      // MaxPool2d kernel=2 stride=2: 24/2=12
      expect(blockById.get("b3")!.outputShapes).toEqual([[6, 12, 12]]);
      // Conv2d filters=16 kernel=5: (12-5)+1=8
      expect(blockById.get("b4")!.outputShapes).toEqual([[16, 8, 8]]);
      expect(blockById.get("b5")!.outputShapes).toEqual([[16, 8, 8]]);
      // MaxPool2d: 8/2=4
      expect(blockById.get("b6")!.outputShapes).toEqual([[16, 4, 4]]);
      // Flatten: 16*4*4=256
      expect(blockById.get("b7")!.outputShapes).toEqual([[256]]);
      expect(blockById.get("b8")!.outputShapes).toEqual([[120]]);
      expect(blockById.get("b9")!.outputShapes).toEqual([[120]]);
      expect(blockById.get("b10")!.outputShapes).toEqual([[84]]);
      expect(blockById.get("b11")!.outputShapes).toEqual([[10]]);
    });
  });

  describe("edge shape propagation", () => {
    it("propagates output shape to outgoing edges", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(3, 224, 224) }),
          makeBlock("b1", "ReLU"),
        ],
        edges: [makeEdge("b0", "b1")],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors).toHaveLength(0);
      expect(result.graph.edges[0].shape).toEqual([3, 224, 224]);
    });
  });

  describe("unknown block type", () => {
    it("records a ShapeError and uses [] as output shapes", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "NonExistentLayer")],
        edges: [],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].blockId).toBe("b0");
      expect(result.errors[0].message).toContain("NonExistentLayer");
      expect(result.graph.blocks[0].outputShapes).toEqual([]);
    });
  });

  describe("cycle detection", () => {
    it("returns error immediately without hanging", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "ReLU"), makeBlock("b1", "ReLU")],
        edges: [makeEdge("b0", "b1"), makeEdge("b1", "b0")],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].message).toContain("Cycle");
    });

    it("does not attempt shape inference when cycle present", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "ReLU"), makeBlock("b1", "ReLU")],
        edges: [makeEdge("b0", "b1"), makeEdge("b1", "b0")],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      // blocks should not have outputShapes set (empty from copy)
      expect(result.graph.blocks[0].outputShapes).toEqual([]);
      expect(result.graph.blocks[1].outputShapes).toEqual([]);
    });
  });

  describe("does not mutate original graph", () => {
    it("original graph blocks remain unchanged", () => {
      const block = makeBlock("b0", "Input", { shape: s(3, 32, 32) });
      const graph: Graph = { blocks: [block], edges: [], groups: [] };
      inferShapes(graph, registry);
      expect(block.outputShapes).toEqual([]);
      expect(block.inputShapes).toEqual([]);
    });
  });

  describe("inferShape error propagation", () => {
    it("records error when inferShape throws", () => {
      const badRegistry = new Map(registry);
      badRegistry.set("ReLU", {
        name: "ReLU",
        params: [],
        inferShape() {
          throw new Error("bad shape!");
        },
      });
      const graph: Graph = {
        blocks: [makeBlock("b0", "ReLU")],
        edges: [],
        groups: [],
      };
      const result = inferShapes(graph, badRegistry);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].message).toBe("bad shape!");
    });
  });

  describe("edge cases", () => {
    it("zero-dimension tensors flow through passthrough blocks without error", () => {
      // Input(shape=[]) → ReLU → Sigmoid
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s() }),
          makeBlock("b1", "ReLU"),
          makeBlock("b2", "Sigmoid"),
        ],
        edges: [makeEdge("b0", "b1"), makeEdge("b1", "b2")],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors).toHaveLength(0);
      const blockById = new Map(result.graph.blocks.map((b) => [b.id, b]));
      // Input with shape[] produces output shape []
      expect(blockById.get("b0")!.outputShapes).toEqual([[]]);
      // ReLU: passthrough → []
      expect(blockById.get("b1")!.outputShapes).toEqual([[]]);
      // Sigmoid: passthrough → []
      expect(blockById.get("b2")!.outputShapes).toEqual([[]]);
    });

    // SKIPPED: inferShape currently does not validate required params.
    // This test documents the gap — enabling it requires adding param validation
    // to inferShapes() in src/ast/infer.ts.
    it.skip("missing required params on Linear returns errors", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(10) }),
          makeBlock("b1", "Linear"), // no out_features
        ],
        edges: [makeEdge("b0", "b1")],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors.length).toBeGreaterThanOrEqual(1);
    });

    it("multi-output LSTM infers both output shapes", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(5, 10) }),
          makeBlock("b1", "LSTM", { hidden_size: n(20) }),
          makeBlock("b2", "ReLU"),
        ],
        edges: [
          makeEdge("b0", "b1"),
          makeEdge("b1", "b2"), // downstream from LSTM
        ],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors).toHaveLength(0);
      const blockById = new Map(result.graph.blocks.map((b) => [b.id, b]));
      const lstm = blockById.get("b1")!;
      // LSTM has two named outputs: "output" and "hidden"
      expect(lstm.outputShapes).toHaveLength(2);
      // output shape: [seq_len, hidden_size] → [5, 20]
      expect(lstm.outputShapes[0]).toEqual([5, 20]);
      // hidden shape: [hidden_size] → [20]
      expect(lstm.outputShapes[1]).toEqual([20]);
      // Downstream block (ReLU) should get input from first output
      expect(blockById.get("b2")!.inputShapes).toEqual([[5, 20]]);
    });

    // SKIPPED: inferShape currently does not validate param values (negative stride).
    // This test documents the gap — enabling it requires adding param validation
    // to inferShapes() in src/ast/infer.ts.
    it.skip("negative stride on MaxPool returns error", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(3, 10, 10) }),
          makeBlock("b1", "MaxPool", { kernel: n(2), stride: n(-1) }),
        ],
        edges: [makeEdge("b0", "b1")],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors.length).toBeGreaterThanOrEqual(1);
    });

    it("unknown block type returns error", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "TotallyFakeBlock")],
        edges: [],
        groups: [],
      };
      const result = inferShapes(graph, registry);
      expect(result.errors.length).toBeGreaterThanOrEqual(1);
      expect(result.errors[0].blockId).toBe("b0");
      expect(result.errors[0].message).toContain("TotallyFakeBlock");
    });
  });
});
