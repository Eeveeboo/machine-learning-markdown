import type { SourceLoc } from "../ast/nodes.js";

export type TokenType =
  | "ARROW"
  | "LBRACKET"
  | "RBRACKET"
  | "LPAREN"
  | "RPAREN"
  | "COMMA"
  | "EQUALS"
  | "GT"
  | "GROUP_OPEN"
  | "GROUP_CLOSE"
  | "IDENT"
  | "NUMBER"
  | "STRING"
  | "BOOL"
  | "NEWLINE"
  | "COMMENT"
  | "EOF";

export interface Token {
  type: TokenType;
  value: string;
  loc: SourceLoc;
}

export function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0;
  let line = 1;
  let col = 1;

  function loc(): SourceLoc {
    return { line, col, offset: pos };
  }

  function advance(n = 1): string {
    const s = source.slice(pos, pos + n);
    for (const ch of s) {
      if (ch === "\n") { line++; col = 1; }
      else { col++; }
      pos++;
    }
    return s;
  }

  function peek(n = 1): string {
    return source.slice(pos, pos + n);
  }

  // Track whether the last non-whitespace/non-newline token before current
  // position allows a unary minus (i.e., preceded by = or ( or ,)
  function allowsUnaryMinus(): boolean {
    for (let i = tokens.length - 1; i >= 0; i--) {
      const t = tokens[i];
      if (t.type === "NEWLINE" || t.type === "COMMENT") continue;
      return t.type === "EQUALS" || t.type === "LPAREN" || t.type === "COMMA";
    }
    return true;
  }

  let lastNewlineEmitted = false;

  while (pos < source.length) {
    // Skip spaces and tabs
    if (peek() === " " || peek() === "\t") {
      advance();
      continue;
    }

    // Newline handling — collapse consecutive newlines
    if (peek() === "\n" || peek() === "\r") {
      const l = loc();
      // consume all consecutive whitespace lines
      while (pos < source.length && (peek() === "\n" || peek() === "\r" || peek() === " " || peek() === "\t")) {
        advance();
      }
      if (!lastNewlineEmitted) {
        tokens.push({ type: "NEWLINE", value: "\n", loc: l });
        lastNewlineEmitted = true;
      }
      continue;
    }

    lastNewlineEmitted = false;

    // Comment
    if (peek() === "#") {
      const l = loc();
      advance(); // skip #
      let text = "";
      while (pos < source.length && peek() !== "\n" && peek() !== "\r") {
        text += advance();
      }
      tokens.push({ type: "COMMENT", value: text.trim(), loc: l });
      continue;
    }

    // Two-char tokens
    if (peek(2) === "->") {
      const l = loc();
      advance(2);
      tokens.push({ type: "ARROW", value: "->", loc: l });
      continue;
    }
    if (peek(2) === "[[") {
      const l = loc();
      advance(2);
      tokens.push({ type: "GROUP_OPEN", value: "[[", loc: l });
      continue;
    }
    if (peek(2) === "]]") {
      const l = loc();
      advance(2);
      tokens.push({ type: "GROUP_CLOSE", value: "]]", loc: l });
      continue;
    }

    // Single-char tokens
    const ch = peek();
    const l = loc();

    if (ch === "[") { advance(); tokens.push({ type: "LBRACKET", value: "[", loc: l }); continue; }
    if (ch === "]") { advance(); tokens.push({ type: "RBRACKET", value: "]", loc: l }); continue; }
    if (ch === "(") { advance(); tokens.push({ type: "LPAREN", value: "(", loc: l }); continue; }
    if (ch === ")") { advance(); tokens.push({ type: "RPAREN", value: ")", loc: l }); continue; }
    if (ch === ",") { advance(); tokens.push({ type: "COMMA", value: ",", loc: l }); continue; }
    if (ch === "=") { advance(); tokens.push({ type: "EQUALS", value: "=", loc: l }); continue; }
    if (ch === ">") { advance(); tokens.push({ type: "GT", value: ">", loc: l }); continue; }

    // String
    if (ch === '"') {
      advance(); // opening quote
      let value = "";
      while (pos < source.length && peek() !== '"') {
        if (peek() === "\\") {
          advance();
          const esc = advance();
          if (esc === "n") value += "\n";
          else if (esc === "t") value += "\t";
          else if (esc === "r") value += "\r";
          else value += esc;
        } else {
          value += advance();
        }
      }
      advance(); // closing quote
      tokens.push({ type: "STRING", value, loc: l });
      continue;
    }

    // Number (possibly negative)
    const isNeg = ch === "-" && /\d/.test(source[pos + 1] ?? "") && allowsUnaryMinus();
    if (isNeg || /\d/.test(ch)) {
      let num = "";
      if (isNeg) { num += advance(); } // consume '-'
      while (pos < source.length && /\d/.test(peek())) {
        num += advance();
      }
      if (peek() === "." && /\d/.test(source[pos + 1] ?? "")) {
        num += advance(); // '.'
        while (pos < source.length && /\d/.test(peek())) {
          num += advance();
        }
      }
      tokens.push({ type: "NUMBER", value: num, loc: l });
      continue;
    }

    // Identifier / bool / keyword
    if (/[a-zA-Z_]/.test(ch)) {
      let ident = "";
      while (pos < source.length && /[a-zA-Z0-9_]/.test(peek())) {
        ident += advance();
      }
      if (ident === "true" || ident === "false") {
        tokens.push({ type: "BOOL", value: ident, loc: l });
      } else {
        tokens.push({ type: "IDENT", value: ident, loc: l });
      }
      continue;
    }

    // Unknown character — skip with a warning (could throw instead)
    advance();
  }

  tokens.push({ type: "EOF", value: "", loc: loc() });
  return tokens;
}
