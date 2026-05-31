import { describe, it, expect } from "vitest";
import { tokenize } from "../../src/parser/tokenizer.js";
import type { TokenType } from "../../src/parser/tokenizer.js";

function types(src: string): TokenType[] {
  return tokenize(src).map((t) => t.type);
}

function values(src: string): string[] {
  return tokenize(src).map((t) => t.value);
}

describe("tokenizer — single tokens", () => {
  it("ARROW", () => expect(types("->")).toEqual(["ARROW", "EOF"]));
  it("LBRACKET", () => expect(types("[")).toEqual(["LBRACKET", "EOF"]));
  it("RBRACKET", () => expect(types("]")).toEqual(["RBRACKET", "EOF"]));
  it("LPAREN", () => expect(types("(")).toEqual(["LPAREN", "EOF"]));
  it("RPAREN", () => expect(types(")")).toEqual(["RPAREN", "EOF"]));
  it("COMMA", () => expect(types(",")).toEqual(["COMMA", "EOF"]));
  it("EQUALS", () => expect(types("=")).toEqual(["EQUALS", "EOF"]));
  it("GT", () => expect(types(">")).toEqual(["GT", "EOF"]));
  it("GROUP_OPEN", () => expect(types("[[")).toEqual(["GROUP_OPEN", "EOF"]));
  it("GROUP_CLOSE", () => expect(types("]]")).toEqual(["GROUP_CLOSE", "EOF"]));
  it("IDENT", () => expect(types("Conv2d")).toEqual(["IDENT", "EOF"]));
  it("BOOL true", () => expect(types("true")).toEqual(["BOOL", "EOF"]));
  it("BOOL false", () => expect(types("false")).toEqual(["BOOL", "EOF"]));
  it("NUMBER int", () => expect(types("42")).toEqual(["NUMBER", "EOF"]));
  it("NUMBER float", () => expect(types("3.14")).toEqual(["NUMBER", "EOF"]));
  it("STRING", () => expect(types('"hello"')).toEqual(["STRING", "EOF"]));
  it("COMMENT", () => expect(types("# a comment")).toEqual(["COMMENT", "EOF"]));
  it("EOF on empty", () => expect(types("")).toEqual(["EOF"]));
});

describe("tokenizer — values", () => {
  it("STRING strips quotes", () => expect(values('"hello world"')[0]).toBe("hello world"));
  it("STRING escape \\n", () => expect(values('"a\\nb"')[0]).toBe("a\nb"));
  it("BOOL value", () => expect(values("true")[0]).toBe("true"));
  it("NUMBER negative via =", () => {
    const toks = tokenize("x=-1");
    expect(toks[2].type).toBe("NUMBER");
    expect(toks[2].value).toBe("-1");
  });
  it("NUMBER negative via (", () => {
    const toks = tokenize("f(-2)");
    expect(toks[2].type).toBe("NUMBER");
    expect(toks[2].value).toBe("-2");
  });
  it("NUMBER negative via ,", () => {
    const toks = tokenize("f(1,-2)");
    expect(toks[4].type).toBe("NUMBER");
    expect(toks[4].value).toBe("-2");
  });
  it("COMMENT value strips leading space", () => {
    const toks = tokenize("# hello there");
    expect(toks[0].value).toBe("hello there");
  });
  it("IDENT with digits and underscore", () => {
    const toks = tokenize("my_block_2");
    expect(toks[0].type).toBe("IDENT");
    expect(toks[0].value).toBe("my_block_2");
  });
});

describe("tokenizer — source locations", () => {
  it("first token at line 1 col 1", () => {
    const toks = tokenize("Input");
    expect(toks[0].loc).toEqual({ line: 1, col: 1, offset: 0 });
  });
  it("token after newline gets correct line", () => {
    const toks = tokenize("A\nB");
    const B = toks.find((t) => t.value === "B")!;
    expect(B.loc.line).toBe(2);
    expect(B.loc.col).toBe(1);
  });
  it("ARROW loc", () => {
    const toks = tokenize("A -> B");
    const arrow = toks.find((t) => t.type === "ARROW")!;
    expect(arrow.loc.col).toBe(3);
  });
});

describe("tokenizer — newlines", () => {
  it("newline is significant", () => expect(types("A\nB")).toContain("NEWLINE"));
  it("consecutive newlines collapse to one", () => {
    const toks = tokenize("A\n\n\nB");
    const newlines = toks.filter((t) => t.type === "NEWLINE");
    expect(newlines.length).toBe(1);
  });
  it("spaces and tabs not emitted as newline", () => {
    expect(types("A B")).not.toContain("NEWLINE");
  });
  it("comment does not consume its trailing newline", () => {
    const toks = tokenize("# comment\nA");
    expect(toks[1].type).toBe("NEWLINE");
    expect(toks[2].type).toBe("IDENT");
  });
});

describe("tokenizer — basic block", () => {
  it("Input()", () => {
    expect(types("Input ()")).toEqual(["IDENT", "LPAREN", "RPAREN", "EOF"]);
  });
  it("Conv2d with params", () => {
    const toks = tokenize("Conv2d (kernel_size=5, filters=6)");
    expect(toks[0]).toMatchObject({ type: "IDENT", value: "Conv2d" });
    expect(toks[2]).toMatchObject({ type: "IDENT", value: "kernel_size" });
    expect(toks[3]).toMatchObject({ type: "EQUALS" });
    expect(toks[4]).toMatchObject({ type: "NUMBER", value: "5" });
    expect(toks[5]).toMatchObject({ type: "COMMA" });
    expect(toks[6]).toMatchObject({ type: "IDENT", value: "filters" });
    expect(toks[7]).toMatchObject({ type: "EQUALS" });
    expect(toks[8]).toMatchObject({ type: "NUMBER", value: "6" });
    expect(toks[9]).toMatchObject({ type: "RPAREN" });
  });
});

describe("tokenizer — arrow chain", () => {
  it("A -> B -> C", () => {
    expect(types("A -> B -> C")).toEqual([
      "IDENT", "ARROW", "IDENT", "ARROW", "IDENT", "EOF",
    ]);
  });
});

describe("tokenizer — tensor naming", () => {
  it("-> [a, b]", () => {
    expect(types("-> [a, b]")).toEqual([
      "ARROW", "LBRACKET", "IDENT", "COMMA", "IDENT", "RBRACKET", "EOF",
    ]);
  });
});

describe("tokenizer — tensor join", () => {
  it("[a, b] -> Block", () => {
    expect(types("[a, b] -> Block")).toEqual([
      "LBRACKET", "IDENT", "COMMA", "IDENT", "RBRACKET",
      "ARROW", "IDENT", "EOF",
    ]);
  });
});

describe("tokenizer — group", () => {
  it("[[ Name ]]", () => {
    expect(types("[[ Name ]]")).toEqual(["GROUP_OPEN", "IDENT", "GROUP_CLOSE", "EOF"]);
  });
  it("[[ A > B ]]", () => {
    expect(types("[[ A > B ]]")).toEqual([
      "GROUP_OPEN", "IDENT", "GT", "IDENT", "GROUP_CLOSE", "EOF",
    ]);
  });
  it("nested group [[ A > [[ B ]] ]]", () => {
    expect(types("[[ A > [[ B ]] ]]")).toEqual([
      "GROUP_OPEN", "IDENT", "GT",
      "GROUP_OPEN", "IDENT", "GROUP_CLOSE",
      "GROUP_CLOSE", "EOF",
    ]);
  });
});

describe("tokenizer — param types", () => {
  it("string param", () => {
    const toks = tokenize('name="relu"');
    expect(toks[2]).toMatchObject({ type: "STRING", value: "relu" });
  });
  it("bool param", () => {
    const toks = tokenize("bias=true");
    expect(toks[2]).toMatchObject({ type: "BOOL", value: "true" });
  });
  it("bareword param (IDENT after =)", () => {
    const toks = tokenize("act=relu");
    expect(toks[2]).toMatchObject({ type: "IDENT", value: "relu" });
  });
  it("float param", () => {
    const toks = tokenize("lr=0.001");
    expect(toks[2]).toMatchObject({ type: "NUMBER", value: "0.001" });
  });
  it("shape param (tuple)", () => {
    // shape=(1, 28, 28) — parenthesised list
    const toks = tokenize("shape=(1, 28, 28)");
    expect(toks.map((t) => t.type)).toEqual([
      "IDENT", "EQUALS", "LPAREN",
      "NUMBER", "COMMA", "NUMBER", "COMMA", "NUMBER",
      "RPAREN", "EOF",
    ]);
  });
});

describe("tokenizer — LeNet example", () => {
  const src = `Input (shape=(1, 28, 28))
    -> Conv2d (kernel_size=5, filters=6)    -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)`;

  it("starts with IDENT Input", () => {
    const toks = tokenize(src);
    expect(toks[0]).toMatchObject({ type: "IDENT", value: "Input" });
  });

  it("has ARROW tokens", () => {
    const toks = tokenize(src);
    const arrows = toks.filter((t) => t.type === "ARROW");
    expect(arrows.length).toBe(3);
  });

  it("shape param parsed correctly", () => {
    const toks = tokenize(src);
    const shapeEq = toks.findIndex((t) => t.value === "shape");
    expect(toks[shapeEq + 1].type).toBe("EQUALS");
    expect(toks[shapeEq + 2].type).toBe("LPAREN");
    expect(toks[shapeEq + 3]).toMatchObject({ type: "NUMBER", value: "1" });
    expect(toks[shapeEq + 5]).toMatchObject({ type: "NUMBER", value: "28" });
    expect(toks[shapeEq + 7]).toMatchObject({ type: "NUMBER", value: "28" });
    expect(toks[shapeEq + 8].type).toBe("RPAREN");
  });

  it("all block names present", () => {
    const toks = tokenize(src);
    const idents = toks.filter((t) => t.type === "IDENT").map((t) => t.value);
    expect(idents).toContain("Input");
    expect(idents).toContain("Conv2d");
    expect(idents).toContain("Tanh");
    expect(idents).toContain("AvgPool");
  });

  it("newlines emitted (one per continuation line)", () => {
    const toks = tokenize(src);
    const newlines = toks.filter((t) => t.type === "NEWLINE");
    // Two newlines in source → two NEWLINE tokens
    expect(newlines.length).toBe(2);
  });

  it("ends with EOF", () => {
    const toks = tokenize(src);
    expect(toks[toks.length - 1].type).toBe("EOF");
  });
});
