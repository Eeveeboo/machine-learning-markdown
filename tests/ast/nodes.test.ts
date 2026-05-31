import { describe, it, expect } from "vitest";
import type {
  SourceLoc,
  BlockDecl,
  TensorName,
  TensorJoin,
  GroupDecl,
  Comment,
  ASTNode,
  ParamValue,
  Param,
} from "../../src/ast/index.js";

const loc: SourceLoc = { line: 1, col: 0, offset: 0 };

describe("AST node construction", () => {
  it("constructs a BlockDecl", () => {
    const param: Param = {
      name: "out_features",
      value: { kind: "number", value: 128, loc } satisfies ParamValue,
      loc,
    };
    const node: BlockDecl = {
      kind: "block",
      blockType: "Linear",
      params: [param],
      loc,
    };
    expect(node.kind).toBe("block");
    expect(node.blockType).toBe("Linear");
    expect(node.params[0]?.name).toBe("out_features");
  });

  it("constructs a TensorName", () => {
    const node: TensorName = { kind: "tensor_name", names: ["x"], loc };
    expect(node.kind).toBe("tensor_name");
    expect(node.names).toEqual(["x"]);
  });

  it("constructs a TensorJoin (fork into two names)", () => {
    const target: BlockDecl = { kind: "block", blockType: "ReLU", params: [], loc };
    const node: TensorJoin = { kind: "tensor_join", sources: ["a", "b"], target, loc };
    expect(node.kind).toBe("tensor_join");
    expect(node.sources).toHaveLength(2);
  });

  it("constructs a GroupDecl", () => {
    const node: GroupDecl = { kind: "group", path: ["Encoder", "Layer"], loc };
    expect(node.kind).toBe("group");
    expect(node.path).toEqual(["Encoder", "Layer"]);
  });

  it("constructs a Comment", () => {
    const node: Comment = { kind: "comment", text: "hello world", loc };
    expect(node.kind).toBe("comment");
  });

  it("ParamValue covers all variants", () => {
    const vals: ParamValue[] = [
      { kind: "number", value: 1, loc },
      { kind: "string", value: "relu", loc },
      { kind: "bool", value: true, loc },
      { kind: "bareword", value: "same", loc },
      { kind: "shape", dims: [3, 224, 224], loc },
      { kind: "list", items: [{ kind: "number", value: 2, loc }], loc },
    ];
    expect(vals).toHaveLength(6);
  });

  it("ASTNode discriminated union narrows correctly", () => {
    const nodes: ASTNode[] = [
      { kind: "block", blockType: "Conv2d", params: [], loc },
      { kind: "tensor_name", names: ["y"], loc },
      { kind: "comment", text: "test", loc },
    ];
    for (const n of nodes) {
      expect(n.kind).toBeTruthy();
    }
  });
});
