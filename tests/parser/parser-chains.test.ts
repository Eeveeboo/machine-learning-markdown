import { describe, it, expect } from "vitest";
import { tokenize } from "../../src/parser/tokenizer.js";
import { parse } from "../../src/parser/parser.js";
import type { BlockDecl, TensorName, TensorJoin } from "../../src/ast/nodes.js";

function parseSource(src: string) {
  return parse(tokenize(src));
}

describe("parser chains", () => {
  it("parses a simple one-liner chain: Input -> ReLU -> Output", () => {
    const { nodes, errors } = parseSource("Input -> ReLU -> Output");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(3);
    expect(nodes[0]).toMatchObject({ kind: "block", blockType: "Input" });
    expect(nodes[1]).toMatchObject({ kind: "block", blockType: "ReLU" });
    expect(nodes[2]).toMatchObject({ kind: "block", blockType: "Output" });
  });

  it("parses Block -> [name] (single tensor name)", () => {
    const { nodes, errors } = parseSource("Conv2d(kernel=3) -> [skip]");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(2);
    expect(nodes[0]).toMatchObject({ kind: "block", blockType: "Conv2d" });
    const tn = nodes[1] as TensorName;
    expect(tn.kind).toBe("tensor_name");
    expect(tn.names).toEqual(["skip"]);
  });

  it("parses Block -> [a, b] (fork)", () => {
    const { nodes, errors } = parseSource("ReLU -> [skip, main]");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(2);
    const tn = nodes[1] as TensorName;
    expect(tn.kind).toBe("tensor_name");
    expect(tn.names).toEqual(["skip", "main"]);
  });

  it("parses [a, b] -> Block (join)", () => {
    const { nodes, errors } = parseSource("[skip, main] -> Add");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const tj = nodes[0] as TensorJoin;
    expect(tj.kind).toBe("tensor_join");
    expect(tj.sources).toEqual(["skip", "main"]);
    expect(tj.target).toMatchObject({ kind: "block", blockType: "Add" });
  });

  it("parses Input -> Conv2d(kernel=1) -> ReLU -> [skip]", () => {
    const { nodes, errors } = parseSource("Input -> Conv2d(kernel=1) -> ReLU -> [skip]");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(4);
    expect(nodes[0]).toMatchObject({ kind: "block", blockType: "Input" });
    expect(nodes[1]).toMatchObject({ kind: "block", blockType: "Conv2d" });
    expect(nodes[2]).toMatchObject({ kind: "block", blockType: "ReLU" });
    const tn = nodes[3] as TensorName;
    expect(tn.kind).toBe("tensor_name");
    expect(tn.names).toEqual(["skip"]);
  });

  it("parses ResNet bottleneck pattern (fork + join)", () => {
    const src = [
      "Input -> Conv2d(kernel=1) -> ReLU -> [skip]",
      "Input -> Conv2d(kernel=3) -> ReLU -> [main]",
      "[skip, main] -> Add -> Output",
    ].join("\n");

    const { nodes, errors } = parseSource(src);
    expect(errors).toHaveLength(0);

    // Line 1: Input, Conv2d, ReLU, TensorName[skip]
    expect(nodes[0]).toMatchObject({ kind: "block", blockType: "Input" });
    expect(nodes[1]).toMatchObject({ kind: "block", blockType: "Conv2d" });
    expect(nodes[2]).toMatchObject({ kind: "block", blockType: "ReLU" });
    expect(nodes[3]).toMatchObject({ kind: "tensor_name", names: ["skip"] });

    // Line 2: Input, Conv2d, ReLU, TensorName[main]
    expect(nodes[4]).toMatchObject({ kind: "block", blockType: "Input" });
    expect(nodes[5]).toMatchObject({ kind: "block", blockType: "Conv2d" });
    expect(nodes[6]).toMatchObject({ kind: "block", blockType: "ReLU" });
    expect(nodes[7]).toMatchObject({ kind: "tensor_name", names: ["main"] });

    // Line 3: TensorJoin([skip,main] -> Add), Output
    const tj = nodes[8] as TensorJoin;
    expect(tj.kind).toBe("tensor_join");
    expect(tj.sources).toEqual(["skip", "main"]);
    expect(tj.target).toMatchObject({ kind: "block", blockType: "Add" });
    expect(nodes[9]).toMatchObject({ kind: "block", blockType: "Output" });
  });

  it("parses a standalone block without chain (regression)", () => {
    const { nodes, errors } = parseSource("Conv2d(kernel=3, stride=1)");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    expect(nodes[0]).toMatchObject({ kind: "block", blockType: "Conv2d" });
  });
});
