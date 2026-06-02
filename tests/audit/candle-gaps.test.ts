import { describe, it, expect } from "vitest";
import "../../src/codegen/targets/candle.js";
import "../../src/codegen/targets/pytorch.js";
import "../../src/codegen/targets/keras.js";
import { getTarget } from "../../src/codegen/target.js";
import "../../src/blocks/index.js";
import type { Graph, Block, Edge } from "../../src/ast/graph.js";

const loc = { line: 1, col: 1, offset: 0 };

function makeBlock(
  id: string,
  type: string,
  params: Block["params"] = {},
  inputShapes: number[][] = [],
  outputShapes: number[][] = [],
): Block {
  return { id, type, params, inputShapes, outputShapes, loc };
}

function num(value: number) {
  return { kind: "number" as const, value, loc };
}

function makeGraph(blocks: Block[], edges: Edge[] = [], groups: Graph["groups"] = []): Graph {
  return { blocks, edges, groups };
}

function generateForBlock(type: string, params: Block["params"], inputShape: number[], outputShape: number[]): string {
  const target = getTarget("candle")!;
  const blocks: Block[] = [
    makeBlock("b0", "Input", {}, [], [inputShape]),
    makeBlock("b1", type, params, [inputShape], [outputShape]),
    makeBlock("b2", "Output", {}, [outputShape], []),
  ];
  const edges: Edge[] = [
    { from: "b0", to: "b1" },
    { from: "b1", to: "b2" },
  ];
  const graph = makeGraph(blocks, edges, [{ path: ["TestModel"], blockIds: ["b0", "b1", "b2"] }]);
  return target.generate(graph, new Map())[0].content;
}

describe("candle codegen — known gaps", () => {
  it("Conv3d generates unimplemented! placeholder", () => {
    const content = generateForBlock(
      "Conv3d",
      { filters: num(16), kernel: num(3), stride: num(1), padding: num(0) },
      [1, 3, 8, 8, 8],
      [16, 6, 6, 6],
    );
    expect(content).toContain('unimplemented!("Conv3d not supported in candle")');
  });

  it("TransposedConv2d generates unimplemented! placeholder", () => {
    const content = generateForBlock(
      "TransposedConv2d",
      { filters: num(16), kernel: num(3), stride: num(2), padding: num(0) },
      [1, 3, 8, 8],
      [16, 17, 17],
    );
    expect(content).toContain('unimplemented!("TransposedConv2d not supported in candle")');
  });

  it("PReLU generates .relu() with comment placeholder", () => {
    const content = generateForBlock(
      "PReLU",
      {},
      [1, 3, 8, 8],
      [1, 3, 8, 8],
    );
    expect(content).toContain(".relu()?");
    expect(content).toContain("/* PReLU: learnable slopes not supported in candle codegen */");
  });

  it("AvgPool generates comment placeholder with AvgPool2d mention", () => {
    const content = generateForBlock(
      "AvgPool",
      { kernel: num(2), stride: num(2) },
      [1, 3, 8, 8],
      [3, 4, 4],
    );
    expect(content).toContain("/* AvgPool2d not directly supported in candle_nn::ops */");
  });

  it("RNN is typed as candle_nn::Linear (placeholder)", () => {
    const content = generateForBlock(
      "RNN",
      { hidden_size: num(64) },
      [1, 10, 32],
      [1, 10, 64],
    );
    // RNN uses Linear as a placeholder in candle; check type annotation in struct field
    expect(content).toContain("b1: candle_nn::Linear");
  });
});
