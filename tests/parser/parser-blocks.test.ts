import { describe, it, expect } from "vitest";
import { tokenize } from "../../src/parser/tokenizer.js";
import { parse } from "../../src/parser/parser.js";
import type { BlockDecl, Comment } from "../../src/ast/nodes.js";

function parseSource(src: string) {
  return parse(tokenize(src));
}

describe("parser — block declarations", () => {
  it("parses a bare block with no params", () => {
    const { nodes, errors } = parseSource("ReLU");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(1);
    const n = nodes[0] as BlockDecl;
    expect(n.kind).toBe("block");
    expect(n.blockType).toBe("ReLU");
    expect(n.params).toHaveLength(0);
  });

  it("parses a block with number params", () => {
    const { nodes, errors } = parseSource("Conv2d(kernel=5, filters=6)");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.blockType).toBe("Conv2d");
    expect(n.params).toHaveLength(2);
    expect(n.params[0]).toMatchObject({ name: "kernel", value: { kind: "number", value: 5 } });
    expect(n.params[1]).toMatchObject({ name: "filters", value: { kind: "number", value: 6 } });
  });

  it("parses a block with a float param", () => {
    const { nodes, errors } = parseSource("Dropout(rate=0.5)");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.params[0]).toMatchObject({ name: "rate", value: { kind: "number", value: 0.5 } });
  });

  it("parses a block with a negative number param", () => {
    const { nodes, errors } = parseSource("Foo(x=-3)");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.params[0]).toMatchObject({ name: "x", value: { kind: "number", value: -3 } });
  });

  it("parses a block with a string param", () => {
    const { nodes, errors } = parseSource('Layer(name="encoder")');
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.params[0]).toMatchObject({ name: "name", value: { kind: "string", value: "encoder" } });
  });

  it("parses a block with bool params", () => {
    const { nodes, errors } = parseSource("BN(affine=true, track=false)");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.params[0]).toMatchObject({ name: "affine", value: { kind: "bool", value: true } });
    expect(n.params[1]).toMatchObject({ name: "track", value: { kind: "bool", value: false } });
  });

  it("parses a block with bareword param", () => {
    const { nodes, errors } = parseSource("Conv2d(padding=same)");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.params[0]).toMatchObject({ name: "padding", value: { kind: "bareword", value: "same" } });
  });

  it("parses Input with shape param", () => {
    const { nodes, errors } = parseSource("Input(shape=(1,28,28))");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.blockType).toBe("Input");
    expect(n.params[0]).toMatchObject({ name: "shape", value: { kind: "shape", dims: [1, 28, 28] } });
  });

  it("parses a block with list param", () => {
    const { nodes, errors } = parseSource("MultiScale(filters=[64,128,256])");
    expect(errors).toHaveLength(0);
    const n = nodes[0] as BlockDecl;
    expect(n.params[0].name).toBe("filters");
    const list = n.params[0].value;
    expect(list.kind).toBe("list");
    if (list.kind === "list") {
      expect(list.items).toHaveLength(3);
      expect(list.items[0]).toMatchObject({ kind: "number", value: 64 });
      expect(list.items[1]).toMatchObject({ kind: "number", value: 128 });
      expect(list.items[2]).toMatchObject({ kind: "number", value: 256 });
    }
  });

  it("parses multiple blocks on separate lines", () => {
    const { nodes, errors } = parseSource("Conv2d(filters=32)\nReLU\nMaxPool(size=2)");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(3);
    expect((nodes[0] as BlockDecl).blockType).toBe("Conv2d");
    expect((nodes[1] as BlockDecl).blockType).toBe("ReLU");
    expect((nodes[2] as BlockDecl).blockType).toBe("MaxPool");
  });

  it("parses comments", () => {
    const { nodes, errors } = parseSource("# a comment\nReLU");
    expect(errors).toHaveLength(0);
    expect(nodes).toHaveLength(2);
    expect(nodes[0].kind).toBe("comment");
    expect((nodes[0] as Comment).text).toBe("a comment");
    expect(nodes[1].kind).toBe("block");
  });

  it("skips faulty lines and continues, collecting errors", () => {
    // "=" at the start of a line is not valid — parser should error and skip
    const { nodes, errors } = parseSource("Conv2d(filters=32)\n=bad\nReLU");
    expect(errors.length).toBeGreaterThan(0);
    // Good lines still parsed
    const blocks = nodes.filter((n) => n.kind === "block") as BlockDecl[];
    expect(blocks.map((b) => b.blockType)).toContain("Conv2d");
    expect(blocks.map((b) => b.blockType)).toContain("ReLU");
  });

  it("recovers after a block with missing closing paren", () => {
    const { nodes, errors } = parseSource("Bad(x=1\nReLU");
    expect(errors.length).toBeGreaterThan(0);
    const blocks = nodes.filter((n) => n.kind === "block") as BlockDecl[];
    // ReLU on next line should still parse
    expect(blocks.some((b) => b.blockType === "ReLU")).toBe(true);
  });

  it("records correct loc for block", () => {
    const { nodes } = parseSource("ReLU");
    const n = nodes[0] as BlockDecl;
    expect(n.loc.line).toBe(1);
    expect(n.loc.col).toBe(1);
  });
});
