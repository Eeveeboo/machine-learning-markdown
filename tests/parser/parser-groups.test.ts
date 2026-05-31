import { describe, it, expect } from "vitest";
import { tokenize } from "../../src/parser/tokenizer.js";
import { parse } from "../../src/parser/parser.js";
import type { GroupDecl, Comment, BlockDecl, TensorName, TensorJoin } from "../../src/ast/nodes.js";

function parseSource(src: string) {
  return parse(tokenize(src));
}

describe("parser — groups", () => {
  it("parses a single-level group with no body", () => {
    const { nodes, errors } = parseSource("[[ Encoder ]]");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const g = nodes[0] as GroupDecl;
    expect(g.kind).toBe("group");
    expect(g.path).toEqual(["Encoder"]);
    expect(g.body).toHaveLength(0);
  });

  it("parses a nested group path", () => {
    const { nodes, errors } = parseSource("[[ Encoder > Block1 ]]");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const g = nodes[0] as GroupDecl;
    expect(g.kind).toBe("group");
    expect(g.path).toEqual(["Encoder", "Block1"]);
  });

  it("parses a three-level nested group path", () => {
    const { nodes, errors } = parseSource("[[ A > B > C ]]");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const g = nodes[0] as GroupDecl;
    expect(g.path).toEqual(["A", "B", "C"]);
  });

  it("group body contains block nodes", () => {
    const src = `[[ Encoder ]]
Conv2d (kernel_size=3)
ReLU
`;
    const { nodes, errors } = parseSource(src);
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const g = nodes[0] as GroupDecl;
    expect(g.body).toHaveLength(2);
    expect((g.body[0] as BlockDecl).blockType).toBe("Conv2d");
    expect((g.body[1] as BlockDecl).blockType).toBe("ReLU");
  });

  it("group body ends at next group", () => {
    const src = `[[ GroupA ]]
Conv2d (kernel_size=3)
[[ GroupB ]]
ReLU
`;
    const { nodes, errors } = parseSource(src);
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(2);
    const a = nodes[0] as GroupDecl;
    const b = nodes[1] as GroupDecl;
    expect(a.path).toEqual(["GroupA"]);
    expect(a.body).toHaveLength(1);
    expect(b.path).toEqual(["GroupB"]);
    expect(b.body).toHaveLength(1);
  });
});

describe("parser — comments", () => {
  it("parses a comment line", () => {
    const { nodes, errors } = parseSource("# this is a comment");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const c = nodes[0] as Comment;
    expect(c.kind).toBe("comment");
    expect(c.text).toBe("this is a comment");
  });

  it("preserves comments in group body", () => {
    const src = `[[ Section ]]
# a comment
ReLU
`;
    const { nodes, errors } = parseSource(src);
    expect(errors).toHaveLength(0);
    const g = nodes[0] as GroupDecl;
    expect(g.body[0].kind).toBe("comment");
    expect(g.body[1].kind).toBe("block");
  });
});

describe("parser — multi-line chains", () => {
  it("parses LeNet-style multi-line chain", () => {
    const src = `Input (shape=(1,28,28))
    -> Conv2d (kernel_size=5, filters=6)
    -> Tanh ()`;
    const { nodes, errors } = parseSource(src);
    expect(errors).toHaveLength(0);
    // Input, Conv2d, Tanh — all in chain
    expect(nodes).toHaveLength(3);
    expect((nodes[0] as BlockDecl).blockType).toBe("Input");
    expect((nodes[1] as BlockDecl).blockType).toBe("Conv2d");
    expect((nodes[2] as BlockDecl).blockType).toBe("Tanh");
  });

  it("parses multi-line chain ending in fork", () => {
    const src = `Input (shape=(256,56,56))
    -> [main, skip]`;
    const { nodes, errors } = parseSource(src);
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(2);
    expect((nodes[0] as BlockDecl).kind).toBe("block");
    expect((nodes[1] as TensorName).kind).toBe("tensor_name");
    expect((nodes[1] as TensorName).names).toEqual(["main", "skip"]);
  });
});

describe("parser — PRODUCT.md examples", () => {
  it("parses LeNet-5 without error", () => {
    const src = `Input (shape=(1,28,28))
    -> Conv2d (kernel_size=5, filters=6)    -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)
    -> Conv2d (kernel_size=5, filters=16)   -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)
    -> Flatten () -> Linear (out_features=120) -> Tanh ()
    -> Linear (out_features=84) -> Tanh ()
    -> Linear (out_features=10) -> Softmax ()`;
    const { errors } = parseSource(src);
    expect(errors).toHaveLength(0);
  });

  it("parses ResNet bottleneck without error", () => {
    const src = `[[ ResNet50 > Bottleneck ]]

Input (shape=(256, 56, 56)) -> [main, skip]

main -> Conv2d (kernel_size=1, filters=64)  -> BatchNorm () -> ReLU ()
    -> Conv2d (kernel_size=3, filters=64, padding=same) -> BatchNorm () -> ReLU ()
    -> Conv2d (kernel_size=1, filters=256)  -> BatchNorm () -> [main_out]

skip -> Conv2d (kernel_size=1, filters=256) -> BatchNorm () -> [skip_out]

[main_out, skip_out] -> Add () -> ReLU ()`;
    const { errors } = parseSource(src);
    expect(errors).toHaveLength(0);
  });

  it("parses attention head without error", () => {
    const src = `Input (shape=(512, 64)) -> [query]
Input (shape=(512, 64)) -> [key]
Input (shape=(512, 64)) -> [value]

query -> Linear (out_features=64, bias=false) -> [q_proj]
key   -> Linear (out_features=64, bias=false) -> [k_proj]
value -> Linear (out_features=64, bias=false) -> [v_proj]

[q_proj, k_proj] -> MatMul () -> Mul (scalar=0.125) -> Softmax (dim=-1) -> [attn]

[attn, v_proj] -> MatMul () -> Linear (out_features=64)`;
    const { errors } = parseSource(src);
    expect(errors).toHaveLength(0);
  });
});
