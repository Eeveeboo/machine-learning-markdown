import { describe, it, expect } from "vitest";
import type { Block, Edge, Graph } from "../../src/ast/graph.js";
import type { SourceLoc } from "../../src/ast/nodes.js";

const loc: SourceLoc = { line: 1, col: 0, offset: 0 };

describe("Graph IR", () => {
  it("constructs a simple 2-block graph", () => {
    const conv: Block = {
      id: "block_0",
      type: "Conv2d",
      params: {
        out_channels: { kind: "number", value: 64, loc },
        kernel_size: { kind: "number", value: 3, loc },
      },
      inputShapes: [[3, 224, 224]],
      outputShapes: [[64, 222, 222]],
      loc,
    };

    const relu: Block = {
      id: "block_1",
      type: "ReLU",
      params: {},
      inputShapes: [[64, 222, 222]],
      outputShapes: [[64, 222, 222]],
      loc,
    };

    const edge: Edge = {
      from: "block_0",
      to: "block_1",
      shape: [64, 222, 222],
    };

    const graph: Graph = {
      blocks: [conv, relu],
      edges: [edge],
      groups: [],
    };

    expect(graph.blocks).toHaveLength(2);
    expect(graph.edges).toHaveLength(1);
    expect(graph.edges[0].from).toBe("block_0");
    expect(graph.edges[0].to).toBe("block_1");
    expect(graph.edges[0].shape).toEqual([64, 222, 222]);
    expect(graph.blocks[0].outputShapes[0]).toEqual([64, 222, 222]);
    expect(graph.groups).toHaveLength(0);
  });

  it("supports named tensor edges", () => {
    const edge: Edge = {
      from: "block_0",
      to: "block_1",
      tensorName: "features",
    };
    expect(edge.tensorName).toBe("features");
    expect(edge.shape).toBeUndefined();
  });

  it("supports multi-output blocks (e.g. LSTM)", () => {
    const lstm: Block = {
      id: "block_0",
      type: "LSTM",
      params: { hidden_size: { kind: "number", value: 256, loc } },
      inputShapes: [[32, 100, 512]],
      outputShapes: [[32, 100, 256], [32, 256]],
      loc,
    };
    expect(lstm.outputShapes).toHaveLength(2);
  });

  it("supports groups with hierarchical paths", () => {
    const graph: Graph = {
      blocks: [],
      edges: [],
      groups: [
        { path: ["ResNet50", "Bottleneck"], blockIds: ["block_0", "block_1"] },
      ],
    };
    expect(graph.groups[0].path).toEqual(["ResNet50", "Bottleneck"]);
    expect(graph.groups[0].blockIds).toContain("block_0");
  });
});
