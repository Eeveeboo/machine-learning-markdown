import { describe, it, expect } from "vitest";
import "../../src/blocks/index.js"; // register all blocks
import { registry } from "../../src/blocks/registry.js";
import { lint } from "../../src/lint/index.js";
import type { Graph, Block, Edge } from "../../src/ast/graph.js";

const loc = { line: 1, col: 0, offset: 0 };

function makeBlock(
  id: string,
  type: string,
  params: Record<string, unknown> = {}
): Block {
  return {
    id,
    type,
    params: params as Block["params"],
    inputShapes: [],
    outputShapes: [],
    loc,
  };
}

function makeEdge(from: string, to: string, tensorName?: string): Edge {
  return tensorName ? { from, to, tensorName } : { from, to };
}

function n(value: number): Block["params"][string] {
  return { kind: "number", value, loc };
}

function s(...dims: number[]): Block["params"][string] {
  return { kind: "shape", dims, loc };
}

function emptyGraph(): Graph {
  return { blocks: [], edges: [], groups: [] };
}

describe("lint()", () => {
  it("returns no diagnostics for a clean simple graph", () => {
    const graph: Graph = {
      blocks: [
        makeBlock("b0", "Input", { shape: s(3, 224, 224) }),
        makeBlock("b1", "ReLU"),
      ],
      edges: [makeEdge("b0", "b1")],
      groups: [],
    };
    const diags = lint(graph, registry);
    expect(diags).toHaveLength(0);
  });

  it("returns no diagnostics for empty graph", () => {
    const diags = lint(emptyGraph(), registry);
    expect(diags).toHaveLength(0);
  });

  describe("cycle detection", () => {
    it("catches a simple cycle", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "ReLU"), makeBlock("b1", "ReLU")],
        edges: [makeEdge("b0", "b1"), makeEdge("b1", "b0")],
        groups: [],
      };
      const diags = lint(graph, registry);
      const cycleErrors = diags.filter((d) => d.rule === "cycle");
      expect(cycleErrors.length).toBeGreaterThan(0);
      expect(cycleErrors[0].severity).toBe("error");
    });

    it("catches a self-loop", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "ReLU")],
        edges: [makeEdge("b0", "b0")],
        groups: [],
      };
      const diags = lint(graph, registry);
      const cycleErrors = diags.filter((d) => d.rule === "cycle");
      expect(cycleErrors.length).toBeGreaterThan(0);
    });
  });

  describe("shape mismatch", () => {
    it("catches Add with incompatible input shapes", () => {
      // Add requires all inputs to have the same shape
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(3, 32, 32) }),
          makeBlock("b1", "Input", { shape: s(3, 64, 64) }),
          makeBlock("b2", "Add"),
        ],
        edges: [makeEdge("b0", "b2"), makeEdge("b1", "b2")],
        groups: [],
      };
      const diags = lint(graph, registry);
      const shapeErrors = diags.filter((d) => d.rule === "shape-mismatch");
      expect(shapeErrors.length).toBeGreaterThan(0);
      expect(shapeErrors[0].severity).toBe("error");
    });

    it("accepts Add with compatible input shapes", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(3, 32, 32) }),
          makeBlock("b1", "Input", { shape: s(3, 32, 32) }),
          makeBlock("b2", "Add"),
        ],
        edges: [makeEdge("b0", "b2"), makeEdge("b1", "b2")],
        groups: [],
      };
      const diags = lint(graph, registry);
      const shapeErrors = diags.filter((d) => d.rule === "shape-mismatch");
      expect(shapeErrors).toHaveLength(0);
    });
  });

  describe("undefined tensor references", () => {
    it("catches reference to undefined tensor name", () => {
      const graph: Graph = {
        blocks: [
          Object.assign(makeBlock("b0", "Add"), {
            joinSources: ["nonexistent_tensor"],
          }),
        ],
        edges: [],
        groups: [],
      };
      const diags = lint(graph, registry);
      const refErrors = diags.filter((d) => d.rule === "undefined-tensor-ref");
      expect(refErrors.length).toBeGreaterThan(0);
      expect(refErrors[0].severity).toBe("error");
      expect(refErrors[0].message).toContain("nonexistent_tensor");
    });

    it("does not flag a defined tensor reference", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(3, 32, 32) }),
          Object.assign(makeBlock("b1", "ReLU"), {
            joinSources: ["my_tensor"],
          }),
        ],
        edges: [makeEdge("b0", "b1", "my_tensor")],
        groups: [],
      };
      const diags = lint(graph, registry);
      const refErrors = diags.filter((d) => d.rule === "undefined-tensor-ref");
      expect(refErrors).toHaveLength(0);
    });
  });

  describe("duplicate tensor names", () => {
    it("catches duplicate tensor names from different source blocks", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(1) }),
          makeBlock("b1", "ReLU"),
          makeBlock("b2", "ReLU"),
          makeBlock("b3", "Add"),
        ],
        edges: [
          makeEdge("b0", "b2", "feat"),
          makeEdge("b1", "b3", "feat"), // same name, DIFFERENT source block → duplicate!
          makeEdge("b2", "b3"),
        ],
        groups: [],
      };
      const diags = lint(graph, registry);
      const dupErrors = diags.filter((d) => d.rule === "duplicate-tensor-name");
      expect(dupErrors.length).toBeGreaterThan(0);
      expect(dupErrors[0].severity).toBe("error");
      expect(dupErrors[0].message).toContain("feat");
    });

    it("allows same tensor name from same source block (fan-out)", () => {
      const graph: Graph = {
        blocks: [
          makeBlock("b0", "Input", { shape: s(1) }),
          makeBlock("b1", "ReLU"),
          makeBlock("b2", "ReLU"),
          makeBlock("b3", "Add"),
        ],
        edges: [
          makeEdge("b0", "b1", "feat"),
          makeEdge("b0", "b2", "feat"), // same source → valid fan-out
          makeEdge("b1", "b3"),
          makeEdge("b2", "b3"),
        ],
        groups: [],
      };
      const diags = lint(graph, registry);
      const dupErrors = diags.filter((d) => d.rule === "duplicate-tensor-name");
      expect(dupErrors.length).toBe(0);
    });
  });

  describe("missing plugin blocks", () => {
    it("catches unknown block type", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "NonExistentBlock123")],
        edges: [],
        groups: [],
      };
      const diags = lint(graph, registry);
      const pluginErrors = diags.filter((d) => d.rule === "missing-plugin");
      expect(pluginErrors.length).toBeGreaterThan(0);
      expect(pluginErrors[0].severity).toBe("error");
      expect(pluginErrors[0].message).toContain("NonExistentBlock123");
    });

    it("does not flag known block types", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "ReLU")],
        edges: [],
        groups: [],
      };
      const diags = lint(graph, registry);
      const pluginErrors = diags.filter((d) => d.rule === "missing-plugin");
      expect(pluginErrors).toHaveLength(0);
    });
  });

  describe("missing required params", () => {
    it("catches missing required param", () => {
      // Linear requires out_features
      const graph: Graph = {
        blocks: [makeBlock("b0", "Linear")], // missing out_features
        edges: [],
        groups: [],
      };
      const diags = lint(graph, registry);
      const paramErrors = diags.filter((d) => d.rule === "missing-param");
      expect(paramErrors.length).toBeGreaterThan(0);
      expect(paramErrors[0].severity).toBe("error");
      expect(paramErrors[0].message).toContain("out_features");
    });

    it("does not flag blocks with all required params", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "Linear", { out_features: n(256) })],
        edges: [],
        groups: [],
      };
      const diags = lint(graph, registry);
      const paramErrors = diags.filter((d) => d.rule === "missing-param");
      expect(paramErrors).toHaveLength(0);
    });
  });

  describe("unused named tensors", () => {
    it("flags a named tensor that has no downstream block", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "Input", { shape: s(3) })],
        edges: [{ from: "b0", to: "orphan_block", tensorName: "orphan" }],
        groups: [],
      };
      const diags = lint(graph, registry);
      const unusedWarnings = diags.filter((d) => d.rule === "unused-tensor");
      expect(unusedWarnings.length).toBeGreaterThan(0);
      expect(unusedWarnings[0].severity).toBe("warning");
      expect(unusedWarnings[0].message).toContain("orphan");
    });
  });

  describe("LintDiagnostic structure", () => {
    it("diagnostic has all required fields", () => {
      const graph: Graph = {
        blocks: [makeBlock("b0", "NonExistentBlock123")],
        edges: [],
        groups: [],
      };
      const diags = lint(graph, registry);
      expect(diags.length).toBeGreaterThan(0);
      const d = diags[0];
      expect(d).toHaveProperty("severity");
      expect(d).toHaveProperty("message");
      expect(d).toHaveProperty("loc");
      expect(d).toHaveProperty("rule");
      expect(["error", "warning"]).toContain(d.severity);
    });
  });
});
