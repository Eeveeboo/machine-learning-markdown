import { describe, it, expect } from "vitest";
import { buildGraph } from "../../src/parser/build-graph.js";
import type { ASTNode, BlockDecl } from "../../src/ast/nodes.js";

const loc = { line: 1, col: 1, offset: 0 };

function block(blockType: string): BlockDecl {
  return { kind: "block", blockType, params: [], loc };
}

describe("buildGraph", () => {
  it("empty input produces empty graph", () => {
    const g = buildGraph([]);
    expect(g.blocks).toHaveLength(0);
    expect(g.edges).toHaveLength(0);
    expect(g.groups).toHaveLength(0);
  });

  it("skips comments", () => {
    const nodes: ASTNode[] = [
      { kind: "comment", text: "# hello", loc },
      block("Conv2d"),
    ];
    const g = buildGraph(nodes);
    expect(g.blocks).toHaveLength(1);
    expect(g.edges).toHaveLength(0);
  });

  it("assigns unique ids per type", () => {
    const nodes: ASTNode[] = [block("Conv2d"), block("Conv2d"), block("ReLU")];
    const g = buildGraph(nodes);
    expect(g.blocks.map((b) => b.id)).toEqual(["Conv2d_0", "Conv2d_1", "ReLU_0"]);
  });

  it("LeNet-style linear chain: head-to-tail edges", () => {
    const nodes: ASTNode[] = [
      block("Conv2d"),
      block("ReLU"),
      block("MaxPool2d"),
      block("Linear"),
    ];
    const g = buildGraph(nodes);
    expect(g.blocks).toHaveLength(4);
    expect(g.edges).toEqual([
      { from: "Conv2d_0", to: "ReLU_0" },
      { from: "ReLU_0", to: "MaxPool2d_0" },
      { from: "MaxPool2d_0", to: "Linear_0" },
    ]);
  });

  it("converts params to Record<string, ParamValue>", () => {
    const nodes: ASTNode[] = [
      {
        kind: "block",
        blockType: "Conv2d",
        params: [
          {
            name: "in_channels",
            value: { kind: "number", value: 3, loc },
            loc,
          },
        ],
        loc,
      },
    ];
    const g = buildGraph(nodes);
    expect(g.blocks[0].params["in_channels"]).toEqual({
      kind: "number",
      value: 3,
      loc,
    });
  });

  describe("ResNet fork/join", () => {
    it("tensor_name saves current tail under name", () => {
      const nodes: ASTNode[] = [
        block("Conv2d"),
        { kind: "tensor_name", names: ["identity"], loc },
        block("ReLU"),
      ];
      const g = buildGraph(nodes);
      // edges: Conv2d_0 -> ReLU_0
      expect(g.edges).toHaveLength(1);
      expect(g.edges[0]).toEqual({ from: "Conv2d_0", to: "ReLU_0" });
    });

    it("tensor_join wires named tensors into target block", () => {
      // Conv2d -> [save as 'skip']
      // -> BN -> ReLU
      // -> Add(skip)
      const nodes: ASTNode[] = [
        block("Conv2d"),
        { kind: "tensor_name", names: ["skip"], loc },
        block("BatchNorm"),
        block("ReLU"),
        {
          kind: "tensor_join",
          sources: ["skip"],
          target: block("Add"),
          loc,
        },
      ];
      const g = buildGraph(nodes);
      // blocks: Conv2d_0, BatchNorm_0, ReLU_0, Add_0
      expect(g.blocks.map((b) => b.id)).toEqual([
        "Conv2d_0",
        "BatchNorm_0",
        "ReLU_0",
        "Add_0",
      ]);
      // Linear chain edges
      expect(g.edges).toContainEqual({ from: "Conv2d_0", to: "BatchNorm_0" });
      expect(g.edges).toContainEqual({ from: "BatchNorm_0", to: "ReLU_0" });
      // Join edge from skip
      expect(g.edges).toContainEqual({
        from: "Conv2d_0",
        to: "Add_0",
        tensorName: "skip",
      });
    });

    it("tensor_join with multiple sources", () => {
      const nodes: ASTNode[] = [
        block("BranchA"),
        { kind: "tensor_name", names: ["a"], loc },
        block("BranchB"),
        { kind: "tensor_name", names: ["b"], loc },
        {
          kind: "tensor_join",
          sources: ["a", "b"],
          target: block("Concat"),
          loc,
        },
      ];
      const g = buildGraph(nodes);
      // Concat should receive edges from a and b
      expect(g.edges).toContainEqual({
        from: "BranchA_0",
        to: "Concat_0",
        tensorName: "a",
      });
      expect(g.edges).toContainEqual({
        from: "BranchB_0",
        to: "Concat_0",
        tensorName: "b",
      });
    });
  });

  describe("GroupDecl", () => {
    it("flattens group body into blocks", () => {
      const nodes: ASTNode[] = [
        {
          kind: "group",
          path: ["MyGroup"],
          body: [block("Conv2d"), block("ReLU")],
          loc,
        },
      ];
      const g = buildGraph(nodes);
      expect(g.blocks).toHaveLength(2);
      expect(g.groups).toHaveLength(1);
      expect(g.groups[0].path).toEqual(["MyGroup"]);
      expect(g.groups[0].blockIds).toEqual(["Conv2d_0", "ReLU_0"]);
    });

    it("nested groups track membership correctly", () => {
      const nodes: ASTNode[] = [
        {
          kind: "group",
          path: ["Outer"],
          body: [
            block("Conv2d"),
            {
              kind: "group",
              path: ["Outer", "Inner"],
              body: [block("ReLU")],
              loc,
            },
          ],
          loc,
        },
      ];
      const g = buildGraph(nodes);
      const outer = g.groups.find((g) => g.path[0] === "Outer" && g.path.length === 1)!;
      const inner = g.groups.find((g) => g.path[1] === "Inner")!;
      expect(outer.blockIds).toContain("Conv2d_0");
      expect(outer.blockIds).toContain("ReLU_0");
      expect(inner.blockIds).toEqual(["ReLU_0"]);
    });

    it("edges flow across group boundaries", () => {
      const nodes: ASTNode[] = [
        block("Input"),
        {
          kind: "group",
          path: ["Layer1"],
          body: [block("Conv2d")],
          loc,
        },
        block("Output"),
      ];
      const g = buildGraph(nodes);
      expect(g.edges).toContainEqual({ from: "Input_0", to: "Conv2d_0" });
      expect(g.edges).toContainEqual({ from: "Conv2d_0", to: "Output_0" });
    });
  });
});
