import type {
  ASTNode,
  BlockDecl,
  Comment,
  GroupDecl,
  TensorName,
  TensorJoin,
  Param,
  ParamValue,
  NumberVal,
  StringVal,
  BoolVal,
  BarewordVal,
  ShapeVal,
  ListVal,
  SourceLoc,
} from "../ast/nodes.js";
import type { Token } from "./tokenizer.js";

export interface ParseError {
  message: string;
  loc: SourceLoc;
}

export interface ParseResult {
  nodes: ASTNode[];
  errors: ParseError[];
}

export function parse(tokens: Token[]): ParseResult {
  const nodes: ASTNode[] = [];
  const errors: ParseError[] = [];
  let pos = 0;

  function peek(): Token {
    return tokens[pos] ?? tokens[tokens.length - 1];
  }

  function advance(): Token {
    const t = tokens[pos];
    if (pos < tokens.length - 1) pos++;
    return t;
  }

  function eat(type: Token["type"]): Token | null {
    if (peek().type === type) return advance();
    return null;
  }

  function skipToNextLine(): void {
    while (peek().type !== "NEWLINE" && peek().type !== "EOF") {
      advance();
    }
    eat("NEWLINE");
  }

  function parseNumber(tok: Token): number {
    return parseFloat(tok.value);
  }

  function parseParamValue(): ParamValue | null {
    const t = peek();

    // Shape: (1,28,28)
    if (t.type === "LPAREN") {
      const loc = t.loc;
      advance(); // (
      const dims: number[] = [];
      // allow empty tuple
      while (peek().type !== "RPAREN" && peek().type !== "EOF" && peek().type !== "NEWLINE") {
        const numTok = eat("NUMBER");
        if (!numTok) {
          errors.push({ message: `Expected number in shape, got ${peek().type}`, loc: peek().loc });
          return null;
        }
        dims.push(parseNumber(numTok));
        if (!eat("COMMA")) break;
      }
      if (!eat("RPAREN")) {
        errors.push({ message: `Expected ) to close shape`, loc: peek().loc });
        return null;
      }
      return { kind: "shape", dims, loc } satisfies ShapeVal;
    }

    // List: [64,128,256]
    if (t.type === "LBRACKET") {
      const loc = t.loc;
      advance(); // [
      const items: ParamValue[] = [];
      while (peek().type !== "RBRACKET" && peek().type !== "EOF" && peek().type !== "NEWLINE") {
        const val = parseParamValue();
        if (val === null) return null;
        items.push(val);
        if (!eat("COMMA")) break;
      }
      if (!eat("RBRACKET")) {
        errors.push({ message: `Expected ] to close list`, loc: peek().loc });
        return null;
      }
      return { kind: "list", items, loc } satisfies ListVal;
    }

    // Number
    if (t.type === "NUMBER") {
      advance();
      return { kind: "number", value: parseNumber(t), loc: t.loc } satisfies NumberVal;
    }

    // String
    if (t.type === "STRING") {
      advance();
      return { kind: "string", value: t.value, loc: t.loc } satisfies StringVal;
    }

    // Bool
    if (t.type === "BOOL") {
      advance();
      return { kind: "bool", value: t.value === "true", loc: t.loc } satisfies BoolVal;
    }

    // Bareword (IDENT used as a value)
    if (t.type === "IDENT") {
      advance();
      return { kind: "bareword", value: t.value, loc: t.loc } satisfies BarewordVal;
    }

    errors.push({ message: `Unexpected token ${t.type} (${t.value}) in param value`, loc: t.loc });
    return null;
  }

  function parseParam(): Param | null {
    const nameTok = eat("IDENT");
    if (!nameTok) {
      errors.push({ message: `Expected param name, got ${peek().type}`, loc: peek().loc });
      return null;
    }
    if (!eat("EQUALS")) {
      errors.push({ message: `Expected = after param name "${nameTok.value}"`, loc: peek().loc });
      return null;
    }
    const value = parseParamValue();
    if (value === null) return null;
    return { name: nameTok.value, value, loc: nameTok.loc };
  }

  function parseBlockDecl(identTok: Token): BlockDecl | null {
    const params: Param[] = [];

    if (eat("LPAREN")) {
      // parse params — may span multiple lines (continuation with trailing comma)
      while (true) {
        // skip newlines inside param list (multi-line params)
        while (peek().type === "NEWLINE") advance();
        if (peek().type === "RPAREN" || peek().type === "EOF") break;
        const param = parseParam();
        if (param === null) return null;
        params.push(param);
        if (!eat("COMMA")) break;
      }
      if (!eat("RPAREN")) {
        errors.push({ message: `Expected ) to close param list`, loc: peek().loc });
        return null;
      }
    }

    return { kind: "block", blockType: identTok.value, params, loc: identTok.loc };
  }

  // Parse a bracket list: [ name1, name2, ... ] — returns names or null on error
  function parseBracketNames(loc: SourceLoc): string[] | null {
    // caller already consumed LBRACKET
    const names: string[] = [];
    while (peek().type !== "RBRACKET" && peek().type !== "EOF" && peek().type !== "NEWLINE") {
      const nameTok = eat("IDENT");
      if (!nameTok) {
        errors.push({ message: `Expected identifier in bracket list, got ${peek().type}`, loc: peek().loc });
        return null;
      }
      names.push(nameTok.value);
      if (!eat("COMMA")) break;
    }
    if (!eat("RBRACKET")) {
      errors.push({ message: `Expected ] to close tensor name list`, loc: peek().loc });
      return null;
    }
    return names;
  }

  // After parsing a block, handle zero or more `-> ...` continuations.
  // Returns the final block (last block in the chain), and emits intermediate nodes.
  // `pendingSources` is set when this chain started from a TensorJoin ([a,b] ->).
  // `out` receives emitted nodes (either top-level or a group body).
  function parseChainTail(
    firstBlock: BlockDecl,
    pendingSources: string[] | null,
    out: ASTNode[],
  ): void {
    let currentBlock: BlockDecl = firstBlock;
    let sources: string[] | null = pendingSources;

    while (true) {
      // Support multi-line chains: after a NEWLINE, if next token is ARROW, continue
      if (peek().type === "NEWLINE") {
        // Look ahead past the newline to see if next meaningful token is ARROW
        let lookahead = pos + 1;
        // skip over extra newlines
        while (lookahead < tokens.length && tokens[lookahead]?.type === "NEWLINE") lookahead++;
        if (tokens[lookahead]?.type !== "ARROW") break;
        // consume newlines up to the arrow
        while (peek().type === "NEWLINE") advance();
      }

      if (peek().type !== "ARROW") break;
      advance(); // consume ->

      const next = peek();

      if (next.type === "LBRACKET") {
        // Fork: Block -> [a, b]  OR  Block -> [a]
        const loc = next.loc;
        advance(); // [
        const names = parseBracketNames(loc);
        if (names === null) { skipToNextLine(); return; }

        // Emit the current block (possibly as part of a join)
        if (sources !== null) {
          out.push({ kind: "tensor_join", sources, target: currentBlock, loc: currentBlock.loc } satisfies TensorJoin);
          sources = null;
        } else {
          out.push(currentBlock);
        }

        // Emit the TensorName
        out.push({ kind: "tensor_name", names, loc } satisfies TensorName);

        // After a fork we expect end of line (no further chaining from names)
        eat("NEWLINE");
        return;
      }

      if (next.type === "IDENT") {
        // Another block in the chain
        advance();
        const nextBlock = parseBlockDecl(next);
        if (nextBlock === null) { skipToNextLine(); return; }

        // Emit the current block (possibly as join)
        if (sources !== null) {
          out.push({ kind: "tensor_join", sources, target: currentBlock, loc: currentBlock.loc } satisfies TensorJoin);
          sources = null;
        } else {
          out.push(currentBlock);
        }

        currentBlock = nextBlock;
        continue;
      }

      // Unexpected token after ->
      errors.push({ message: `Expected block or [ after ->, got ${next.type}`, loc: next.loc });
      skipToNextLine();
      return;
    }

    // No more arrows — emit the final block
    if (sources !== null) {
      out.push({ kind: "tensor_join", sources, target: currentBlock, loc: currentBlock.loc } satisfies TensorJoin);
    } else {
      out.push(currentBlock);
    }
    eat("NEWLINE");
  }

  // Parse a group declaration: GROUP_OPEN already consumed.
  // Returns the group node (body filled after this call site collects body lines).
  function parseGroupDecl(loc: SourceLoc): GroupDecl | null {
    const path: string[] = [];

    // Expect: IDENT (GT IDENT)* GROUP_CLOSE
    const first = eat("IDENT");
    if (!first) {
      errors.push({ message: `Expected group name after [[`, loc: peek().loc });
      skipToNextLine();
      return null;
    }
    path.push(first.value);

    while (peek().type === "GT") {
      advance(); // consume >
      const seg = eat("IDENT");
      if (!seg) {
        errors.push({ message: `Expected group name segment after >`, loc: peek().loc });
        skipToNextLine();
        return null;
      }
      path.push(seg.value);
    }

    if (!eat("GROUP_CLOSE")) {
      errors.push({ message: `Expected ]] to close group declaration`, loc: peek().loc });
      skipToNextLine();
      return null;
    }

    // optional trailing comment / newline
    eat("COMMENT");
    eat("NEWLINE");

    return { kind: "group", path, body: [], loc } satisfies GroupDecl;
  }

  function parseLine(out: ASTNode[]): void {
    // skip blank lines
    while (eat("NEWLINE")) { /* skip */ }

    const t = peek();

    if (t.type === "EOF") return;

    if (t.type === "COMMENT") {
      advance();
      out.push({ kind: "comment", text: t.value, loc: t.loc } satisfies Comment);
      eat("NEWLINE");
      return;
    }

    // Group declaration: [[ ... ]]
    if (t.type === "GROUP_OPEN") {
      advance();
      const group = parseGroupDecl(t.loc);
      if (group === null) return;

      // Collect body lines until next GROUP_OPEN or EOF
      while (peek().type !== "EOF" && peek().type !== "GROUP_OPEN") {
        // skip blank lines
        while (eat("NEWLINE")) { /* skip */ }
        if (peek().type === "EOF" || peek().type === "GROUP_OPEN") break;
        parseLine(group.body);
      }

      out.push(group);
      return;
    }

    // Join: [a, b] -> Block
    if (t.type === "LBRACKET") {
      const loc = t.loc;
      advance(); // [
      const names = parseBracketNames(loc);
      if (names === null) { skipToNextLine(); return; }

      if (!eat("ARROW")) {
        errors.push({ message: `Expected -> after tensor list`, loc: peek().loc });
        skipToNextLine();
        return;
      }

      const identTok = eat("IDENT");
      if (!identTok) {
        errors.push({ message: `Expected block name after ->, got ${peek().type}`, loc: peek().loc });
        skipToNextLine();
        return;
      }

      const block = parseBlockDecl(identTok);
      if (block === null) { skipToNextLine(); return; }

      parseChainTail(block, names, out);
      return;
    }

    if (t.type === "IDENT") {
      advance();
      const blockDecl = parseBlockDecl(t);
      if (blockDecl === null) {
        skipToNextLine();
        return;
      }

      // Check for chain
      if (peek().type === "ARROW") {
        parseChainTail(blockDecl, null, out);
        return;
      }

      // Check for multi-line chain (NEWLINE followed by ARROW)
      if (peek().type === "NEWLINE") {
        let lookahead = pos + 1;
        while (lookahead < tokens.length && tokens[lookahead]?.type === "NEWLINE") lookahead++;
        if (tokens[lookahead]?.type === "ARROW") {
          parseChainTail(blockDecl, null, out);
          return;
        }
      }

      out.push(blockDecl);
      eat("NEWLINE");
      return;
    }

    // Unknown line start — skip
    errors.push({ message: `Unexpected token ${t.type} at start of line`, loc: t.loc });
    skipToNextLine();
  }

  while (peek().type !== "EOF") {
    parseLine(nodes);
  }

  return { nodes, errors };
}
