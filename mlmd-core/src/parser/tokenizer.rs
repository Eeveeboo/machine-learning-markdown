use crate::ast::nodes::SourceLoc;

// ---------------------------------------------------------------------------
// Token types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Arrow,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
    Equals,
    Gt,
    GroupOpen,
    GroupClose,
    Ident,
    Number,
    String,
    Bool,
    Newline,
    Comment,
    Eof,
}

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
    pub loc: SourceLoc,
}

impl Token {
    pub fn new(token_type: TokenType, value: impl Into<String>, loc: SourceLoc) -> Self {
        Self {
            token_type,
            value: value.into(),
            loc,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Scans the source string and produces a flat vector of tokens.
///
/// Behaviour is intentionally identical to the TypeScript original:
/// - Spaces and tabs are skipped (but not newlines).
/// - Consecutive newlines (including `\r`) are collapsed into a single NEWLINE
///   token (the first one encountered).
/// - Comments start with `#` and run to end-of-line; the `#` and trailing
///   newline are consumed; the stored value is the comment text trimmed.
/// - Two-character tokens (`->`, `[[`, `]]`) are checked before single-character
///   tokens, so `[[` is always GROUP_OPEN rather than two LBRACKETs.
/// - Unary minus detection: a `-` immediately followed by a digit is treated as
///   part of a negative number only when the last semantic token is EQUALS,
///   LPAREN, COMMA, or there is no prior token (beginning of input).  NEWLINE
///   and COMMENT tokens are skipped when making this decision.
/// - Strings support the escapes `\n`, `\t`, `\r`, `\\`, and `\"`.
/// - The identifier scanner also recognises `true` / `false` and emits BOOL.
/// - Unknown characters are silently skipped.
/// - An EOF token is always appended as the final token.
pub fn tokenize(source: &str) -> Vec<Token> {
    let src = source.as_bytes();
    let len = src.len();
    let mut tokens: Vec<Token> = Vec::new();
    let mut pos: usize = 0;
    let mut line: usize = 1;
    let mut col: usize = 1;
    let mut last_newline_emitted = false;

    while pos < len {
        let ch = src[pos];

        // ------------------------------------------------------------------
        // Skip spaces and tabs (but not newlines)
        // ------------------------------------------------------------------
        if ch == b' ' || ch == b'\t' {
            pos += 1;
            col += 1;
            continue;
        }

        // ------------------------------------------------------------------
        // Newline handling — collapse consecutive whitespace lines
        // ------------------------------------------------------------------
        if ch == b'\n' || ch == b'\r' {
            let loc = SourceLoc { line, col, offset: pos };

            // Consume ALL consecutive \n, \r, space, tab
            while pos < len {
                let c = src[pos];
                if c == b'\n' || c == b'\r' || c == b' ' || c == b'\t' {
                    if c == b'\n' {
                        line += 1;
                        col = 1;
                    } else {
                        col += 1;
                    }
                    pos += 1;
                } else {
                    break;
                }
            }

            if !last_newline_emitted {
                tokens.push(Token::new(TokenType::Newline, "\n", loc));
                last_newline_emitted = true;
            }
            continue;
        }

        last_newline_emitted = false;

        // ------------------------------------------------------------------
        // Comment
        // ------------------------------------------------------------------
        if ch == b'#' {
            let loc = SourceLoc { line, col, offset: pos };
            pos += 1; // skip '#'
            col += 1;

            let mut text = String::new();
            while pos < len && src[pos] != b'\n' && src[pos] != b'\r' {
                text.push(src[pos] as char);
                pos += 1;
                col += 1;
            }
            tokens.push(Token::new(TokenType::Comment, text.trim(), loc));
            continue;
        }

        // ------------------------------------------------------------------
        // Two-character tokens
        // ------------------------------------------------------------------
        if pos + 1 < len {
            let two = [src[pos], src[pos + 1]];
            if two == [b'-', b'>'] {
                let loc = SourceLoc { line, col, offset: pos };
                pos += 2;
                col += 2;
                tokens.push(Token::new(TokenType::Arrow, "->", loc));
                continue;
            }
            if two == [b'[', b'['] {
                let loc = SourceLoc { line, col, offset: pos };
                pos += 2;
                col += 2;
                tokens.push(Token::new(TokenType::GroupOpen, "[[", loc));
                continue;
            }
            if two == [b']', b']'] {
                let loc = SourceLoc { line, col, offset: pos };
                pos += 2;
                col += 2;
                tokens.push(Token::new(TokenType::GroupClose, "]]", loc));
                continue;
            }
        }

        // ------------------------------------------------------------------
        // Single-character tokens + string + number + identifier
        //
        // Capture the source location *before* consuming anything so that
        // every branch below can use the same `loc`.
        // ------------------------------------------------------------------
        let loc = SourceLoc { line, col, offset: pos };

        // Single-char tokens
        if ch == b'[' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::LBracket, "[", loc));
            continue;
        }
        if ch == b']' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::RBracket, "]", loc));
            continue;
        }
        if ch == b'(' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::LParen, "(", loc));
            continue;
        }
        if ch == b')' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::RParen, ")", loc));
            continue;
        }
        if ch == b',' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::Comma, ",", loc));
            continue;
        }
        if ch == b'=' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::Equals, "=", loc));
            continue;
        }
        if ch == b'>' {
            pos += 1;
            col += 1;
            tokens.push(Token::new(TokenType::Gt, ">", loc));
            continue;
        }

        // ------------------------------------------------------------------
        // String
        // ------------------------------------------------------------------
        if ch == b'"' {
            pos += 1; // consume opening quote
            col += 1;
            let mut value = String::new();
            while pos < len && src[pos] != b'"' {
                if src[pos] == b'\\' {
                    pos += 1;
                    col += 1;
                    if pos < len {
                        let esc = src[pos];
                        pos += 1;
                        col += 1;
                        match esc {
                            b'n' => value.push('\n'),
                            b't' => value.push('\t'),
                            b'r' => value.push('\r'),
                            _ => value.push(esc as char), // \\, \", etc.
                        }
                    }
                } else {
                    value.push(src[pos] as char);
                    pos += 1;
                    col += 1;
                }
            }
            if pos < len {
                pos += 1; // consume closing quote
                col += 1;
            }
            tokens.push(Token::new(TokenType::String, value, loc));
            continue;
        }

        // ------------------------------------------------------------------
        // Number (possibly negative)
        // ------------------------------------------------------------------
        let is_neg = ch == b'-'
            && pos + 1 < len
            && src[pos + 1].is_ascii_digit()
            && allows_unary_minus(&tokens);

        if is_neg || ch.is_ascii_digit() {
            let mut num = String::new();
            if is_neg {
                num.push('-');
                pos += 1;
                col += 1;
            }
            while pos < len && src[pos].is_ascii_digit() {
                num.push(src[pos] as char);
                pos += 1;
                col += 1;
            }
            // Optional fractional part
            if pos + 1 < len && src[pos] == b'.' && src[pos + 1].is_ascii_digit() {
                num.push('.');
                pos += 1;
                col += 1;
                while pos < len && src[pos].is_ascii_digit() {
                    num.push(src[pos] as char);
                    pos += 1;
                    col += 1;
                }
            }
            tokens.push(Token::new(TokenType::Number, num, loc));
            continue;
        }

        // ------------------------------------------------------------------
        // Identifier / bool
        // ------------------------------------------------------------------
        if ch.is_ascii_alphabetic() || ch == b'_' {
            let mut ident = String::new();
            while pos < len && (src[pos].is_ascii_alphanumeric() || src[pos] == b'_') {
                ident.push(src[pos] as char);
                pos += 1;
                col += 1;
            }
            if ident == "true" || ident == "false" {
                tokens.push(Token::new(TokenType::Bool, ident, loc));
            } else {
                tokens.push(Token::new(TokenType::Ident, ident, loc));
            }
            continue;
        }

        // ------------------------------------------------------------------
        // Unknown character — skip silently (mirrors TS behaviour)
        // ------------------------------------------------------------------
        pos += 1;
        col += 1;
    }

    // Final EOF token
    tokens.push(Token::new(
        TokenType::Eof,
        "",
        SourceLoc { line, col, offset: pos },
    ));

    tokens
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the last *semantic* token (skipping NEWLINE and COMMENT)
/// is EQUALS, LPAREN, COMMA, or there are no prior tokens at all.
fn allows_unary_minus(tokens: &[Token]) -> bool {
    for token in tokens.iter().rev() {
        match token.token_type {
            TokenType::Newline | TokenType::Comment => continue,
            TokenType::Equals | TokenType::LParen | TokenType::Comma => return true,
            _ => return false,
        }
    }
    true // beginning of input
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----------------------------------------------------------

    fn loc(line: usize, col: usize, offset: usize) -> SourceLoc {
        SourceLoc { line, col, offset }
    }

    fn eof_at(line: usize, col: usize, offset: usize) -> Token {
        Token::new(TokenType::Eof, "", loc(line, col, offset))
    }

    fn assert_tokens(source: &str, expected: &[Token]) {
        let result = tokenize(source);
        assert_eq!(
            result.len(),
            expected.len(),
            "token count mismatch for {source:?}\n  got:      {result:#?}\n  expected: {expected:#?}"
        );
        for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, exp,
                "token[{i}] mismatch for {source:?}\n  got:      {result:#?}\n  expected: {expected:#?}"
            );
        }
    }

    // ---- basic block ------------------------------------------------------

    #[test]
    fn test_basic_block() {
        assert_tokens(
            "Conv2d(kernel=5)",
            &[
                Token::new(TokenType::Ident, "Conv2d", loc(1, 1, 0)),
                Token::new(TokenType::LParen, "(", loc(1, 7, 6)),
                Token::new(TokenType::Ident, "kernel", loc(1, 8, 7)),
                Token::new(TokenType::Equals, "=", loc(1, 14, 13)),
                Token::new(TokenType::Number, "5", loc(1, 15, 14)),
                Token::new(TokenType::RParen, ")", loc(1, 16, 15)),
                eof_at(1, 17, 16),
            ],
        );
    }

    // ---- arrow chains -----------------------------------------------------

    #[test]
    fn test_arrow_chain() {
        assert_tokens(
            "Input -> Linear(out=10) -> Output",
            &[
                Token::new(TokenType::Ident, "Input", loc(1, 1, 0)),
                Token::new(TokenType::Arrow, "->", loc(1, 7, 6)),
                Token::new(TokenType::Ident, "Linear", loc(1, 10, 9)),
                Token::new(TokenType::LParen, "(", loc(1, 16, 15)),
                Token::new(TokenType::Ident, "out", loc(1, 17, 16)),
                Token::new(TokenType::Equals, "=", loc(1, 20, 19)),
                Token::new(TokenType::Number, "10", loc(1, 21, 20)),
                Token::new(TokenType::RParen, ")", loc(1, 23, 22)),
                Token::new(TokenType::Arrow, "->", loc(1, 25, 24)),
                Token::new(TokenType::Ident, "Output", loc(1, 28, 27)),
                eof_at(1, 34, 33),
            ],
        );
    }

    // ---- multi-line chains ------------------------------------------------

    #[test]
    fn test_multi_line_chain() {
        assert_tokens(
            "Input\n  -> Linear(out=10)\n  -> Output\n",
            &[
                Token::new(TokenType::Ident, "Input", loc(1, 1, 0)),
                Token::new(TokenType::Newline, "\n", loc(1, 6, 5)),
                Token::new(TokenType::Arrow, "->", loc(2, 3, 8)),
                Token::new(TokenType::Ident, "Linear", loc(2, 6, 11)),
                Token::new(TokenType::LParen, "(", loc(2, 12, 17)),
                Token::new(TokenType::Ident, "out", loc(2, 13, 18)),
                Token::new(TokenType::Equals, "=", loc(2, 16, 21)),
                Token::new(TokenType::Number, "10", loc(2, 17, 22)),
                Token::new(TokenType::RParen, ")", loc(2, 19, 24)),
                Token::new(TokenType::Newline, "\n", loc(2, 20, 25)),
                Token::new(TokenType::Arrow, "->", loc(3, 3, 28)),
                Token::new(TokenType::Ident, "Output", loc(3, 6, 31)),
                Token::new(TokenType::Newline, "\n", loc(3, 12, 37)),
                eof_at(4, 1, 38),
            ],
        );
    }

    // ---- negative numbers -------------------------------------------------

    #[test]
    fn test_negative_numbers() {
        assert_tokens(
            "Conv2d(filters=-1)",
            &[
                Token::new(TokenType::Ident, "Conv2d", loc(1, 1, 0)),
                Token::new(TokenType::LParen, "(", loc(1, 7, 6)),
                Token::new(TokenType::Ident, "filters", loc(1, 8, 7)),
                Token::new(TokenType::Equals, "=", loc(1, 15, 14)),
                Token::new(TokenType::Number, "-1", loc(1, 16, 15)),
                Token::new(TokenType::RParen, ")", loc(1, 18, 17)),
                eof_at(1, 19, 18),
            ],
        );
    }

    #[test]
    fn test_negative_number_in_parens() {
        assert_tokens(
            "scale(-1)",
            &[
                Token::new(TokenType::Ident, "scale", loc(1, 1, 0)),
                Token::new(TokenType::LParen, "(", loc(1, 6, 5)),
                Token::new(TokenType::Number, "-1", loc(1, 7, 6)),
                Token::new(TokenType::RParen, ")", loc(1, 9, 8)),
                eof_at(1, 10, 9),
            ],
        );
    }

    #[test]
    fn test_negative_number_after_comma() {
        assert_tokens(
            "a(x=-1, y=-2)",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::LParen, "(", loc(1, 2, 1)),
                Token::new(TokenType::Ident, "x", loc(1, 3, 2)),
                Token::new(TokenType::Equals, "=", loc(1, 4, 3)),
                Token::new(TokenType::Number, "-1", loc(1, 5, 4)),
                Token::new(TokenType::Comma, ",", loc(1, 7, 6)),
                Token::new(TokenType::Ident, "y", loc(1, 9, 8)),
                Token::new(TokenType::Equals, "=", loc(1, 10, 9)),
                Token::new(TokenType::Number, "-2", loc(1, 11, 10)),
                Token::new(TokenType::RParen, ")", loc(1, 13, 12)),
                eof_at(1, 14, 13),
            ],
        );
    }

    #[test]
    fn test_hyphen_not_unary_minus_after_ident() {
        // A bare '-' after an identifier is NOT a unary minus; the TS tokenizer
        // skips it as an unknown character, so the identifier is not joined.
        assert_tokens(
            "a-b",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Ident, "b", loc(1, 3, 2)),
                eof_at(1, 4, 3),
            ],
        );
    }

    // ---- string params ----------------------------------------------------

    #[test]
    fn test_string_params() {
        assert_tokens(
            r#"Conv2d(padding="same")"#,
            &[
                Token::new(TokenType::Ident, "Conv2d", loc(1, 1, 0)),
                Token::new(TokenType::LParen, "(", loc(1, 7, 6)),
                Token::new(TokenType::Ident, "padding", loc(1, 8, 7)),
                Token::new(TokenType::Equals, "=", loc(1, 15, 14)),
                Token::new(TokenType::String, "same", loc(1, 16, 15)),
                Token::new(TokenType::RParen, ")", loc(1, 22, 21)),
                eof_at(1, 23, 22),
            ],
        );
    }

    #[test]
    fn test_string_escapes() {
        assert_tokens(
            r#"msg("hello\nworld\t!")"#,
            &[
                Token::new(TokenType::Ident, "msg", loc(1, 1, 0)),
                Token::new(TokenType::LParen, "(", loc(1, 4, 3)),
                Token::new(TokenType::String, "hello\nworld\t!", loc(1, 5, 4)),
                Token::new(TokenType::RParen, ")", loc(1, 22, 21)),
                eof_at(1, 23, 22),
            ],
        );
    }

    #[test]
    fn test_string_escape_backslash_quote() {
        assert_tokens(
            r#"x = "a\"b""#,
            &[
                Token::new(TokenType::Ident, "x", loc(1, 1, 0)),
                Token::new(TokenType::Equals, "=", loc(1, 3, 2)),
                Token::new(TokenType::String, "a\"b", loc(1, 5, 4)),
                eof_at(1, 11, 10),
            ],
        );
    }

    // ---- comments ---------------------------------------------------------

    #[test]
    fn test_comment() {
        assert_tokens(
            "Input # this is a comment\nOutput",
            &[
                Token::new(TokenType::Ident, "Input", loc(1, 1, 0)),
                Token::new(TokenType::Comment, "this is a comment", loc(1, 7, 6)),
                Token::new(TokenType::Newline, "\n", loc(1, 26, 25)),
                Token::new(TokenType::Ident, "Output", loc(2, 1, 26)),
                eof_at(2, 7, 32),
            ],
        );
    }

    #[test]
    fn test_comment_at_eof() {
        assert_tokens(
            "x # just a comment",
            &[
                Token::new(TokenType::Ident, "x", loc(1, 1, 0)),
                Token::new(TokenType::Comment, "just a comment", loc(1, 3, 2)),
                eof_at(1, 19, 18),
            ],
        );
    }

    // ---- groups -----------------------------------------------------------

    #[test]
    fn test_groups() {
        assert_tokens(
            "[[ ResNet > Block1 ]]",
            &[
                Token::new(TokenType::GroupOpen, "[[", loc(1, 1, 0)),
                Token::new(TokenType::Ident, "ResNet", loc(1, 4, 3)),
                Token::new(TokenType::Gt, ">", loc(1, 11, 10)),
                Token::new(TokenType::Ident, "Block1", loc(1, 13, 12)),
                Token::new(TokenType::GroupClose, "]]", loc(1, 20, 19)),
                eof_at(1, 22, 21),
            ],
        );
    }

    // ---- tensor joins -----------------------------------------------------

    #[test]
    fn test_tensor_joins() {
        assert_tokens(
            "[a, b] -> Add()",
            &[
                Token::new(TokenType::LBracket, "[", loc(1, 1, 0)),
                Token::new(TokenType::Ident, "a", loc(1, 2, 1)),
                Token::new(TokenType::Comma, ",", loc(1, 3, 2)),
                Token::new(TokenType::Ident, "b", loc(1, 5, 4)),
                Token::new(TokenType::RBracket, "]", loc(1, 6, 5)),
                Token::new(TokenType::Arrow, "->", loc(1, 8, 7)),
                Token::new(TokenType::Ident, "Add", loc(1, 11, 10)),
                Token::new(TokenType::LParen, "(", loc(1, 14, 13)),
                Token::new(TokenType::RParen, ")", loc(1, 15, 14)),
                eof_at(1, 16, 15),
            ],
        );
    }

    // ---- booleans ---------------------------------------------------------

    #[test]
    fn test_booleans() {
        assert_tokens(
            "a=true b=false",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Equals, "=", loc(1, 2, 1)),
                Token::new(TokenType::Bool, "true", loc(1, 3, 2)),
                Token::new(TokenType::Ident, "b", loc(1, 8, 7)),
                Token::new(TokenType::Equals, "=", loc(1, 9, 8)),
                Token::new(TokenType::Bool, "false", loc(1, 10, 9)),
                eof_at(1, 15, 14),
            ],
        );
    }

    // ---- float numbers ----------------------------------------------------

    #[test]
    fn test_float_numbers() {
        assert_tokens(
            "pi=3.14",
            &[
                Token::new(TokenType::Ident, "pi", loc(1, 1, 0)),
                Token::new(TokenType::Equals, "=", loc(1, 3, 2)),
                Token::new(TokenType::Number, "3.14", loc(1, 4, 3)),
                eof_at(1, 8, 7),
            ],
        );
    }

    // ---- edge cases -------------------------------------------------------

    #[test]
    fn test_empty_input() {
        assert_tokens("", &[eof_at(1, 1, 0)]);
    }

    #[test]
    fn test_only_newlines() {
        assert_tokens(
            "\n\n\n",
            &[
                Token::new(TokenType::Newline, "\n", loc(1, 1, 0)),
                eof_at(4, 1, 3),
            ],
        );
    }

    #[test]
    fn test_only_comments() {
        assert_tokens(
            "# first\n# second\n",
            &[
                Token::new(TokenType::Comment, "first", loc(1, 1, 0)),
                Token::new(TokenType::Newline, "\n", loc(1, 8, 7)),
                Token::new(TokenType::Comment, "second", loc(2, 1, 8)),
                Token::new(TokenType::Newline, "\n", loc(2, 9, 16)),
                eof_at(3, 1, 17),
            ],
        );
    }

    #[test]
    fn test_only_comment_no_newline() {
        assert_tokens(
            "# just a comment",
            &[
                Token::new(TokenType::Comment, "just a comment", loc(1, 1, 0)),
                eof_at(1, 17, 16),
            ],
        );
    }

    #[test]
    fn test_mixed_whitespace_newline_collapse() {
        // Spaces at the end of a line, then a newline, more spaces, then text.
        // The spaces before the newline should be consumed as part of the
        // newline block.
        assert_tokens(
            "a   \n   b",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Newline, "\n", loc(1, 5, 4)),
                Token::new(TokenType::Ident, "b", loc(2, 4, 8)),
                eof_at(2, 5, 9),
            ],
        );
    }

    #[test]
    fn test_carriage_return_newline() {
        // \r\n should be treated as a single newline sequence
        assert_tokens(
            "a\r\nb",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Newline, "\n", loc(1, 2, 1)),
                Token::new(TokenType::Ident, "b", loc(2, 1, 3)),
                eof_at(2, 2, 4),
            ],
        );
    }

    #[test]
    fn test_consecutive_newlines_collapsed() {
        // Multiple blank lines should produce only one NEWLINE token
        assert_tokens(
            "a\n\n\nb",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Newline, "\n", loc(1, 2, 1)),
                Token::new(TokenType::Ident, "b", loc(4, 1, 4)),
                eof_at(4, 2, 5),
            ],
        );
    }

    #[test]
    fn test_unknown_characters_skipped() {
        // Characters like @, $, etc. are silently skipped
        assert_tokens(
            "a@b$c",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Ident, "b", loc(1, 3, 2)),
                Token::new(TokenType::Ident, "c", loc(1, 5, 4)),
                eof_at(1, 6, 5),
            ],
        );
    }

    #[test]
    fn test_lone_minus_not_unary() {
        // A minus not followed by a digit is skipped
        assert_tokens(
            "a - b",
            &[
                Token::new(TokenType::Ident, "a", loc(1, 1, 0)),
                Token::new(TokenType::Ident, "b", loc(1, 5, 4)),
                eof_at(1, 6, 5),
            ],
        );
    }

    // ---- round-trip verification against TS reference ----------------------

    /// Helper: produce the same output shape as the TS version for debugging.
    /// The real assertion is via `assert_tokens` above; this test is a
    /// sanity check that the token stream at least has the expected types
    /// in order.
    #[test]
    fn test_types_in_order() {
        let tokens = tokenize("Conv2d(kernel=5)");
        let types: Vec<&str> = tokens.iter().map(|t| match t.token_type {
            TokenType::Ident => "IDENT",
            TokenType::Number => "NUMBER",
            TokenType::String => "STRING",
            TokenType::Bool => "BOOL",
            TokenType::Arrow => "ARROW",
            TokenType::LBracket => "LBRACKET",
            TokenType::RBracket => "RBRACKET",
            TokenType::LParen => "LPAREN",
            TokenType::RParen => "RPAREN",
            TokenType::Comma => "COMMA",
            TokenType::Equals => "EQUALS",
            TokenType::Gt => "GT",
            TokenType::GroupOpen => "GROUP_OPEN",
            TokenType::GroupClose => "GROUP_CLOSE",
            TokenType::Newline => "NEWLINE",
            TokenType::Comment => "COMMENT",
            TokenType::Eof => "EOF",
        }).collect();
        assert_eq!(types, &["IDENT", "LPAREN", "IDENT", "EQUALS", "NUMBER", "RPAREN", "EOF"]);
    }

    #[test]
    fn test_arrow_chain_types() {
        let tokens = tokenize("Input -> Linear(out=10) -> Output");
        let types: Vec<&str> = tokens.iter().map(|t| match t.token_type {
            TokenType::Ident => "IDENT",
            TokenType::Number => "NUMBER",
            TokenType::Arrow => "ARROW",
            TokenType::LParen => "LPAREN",
            TokenType::RParen => "RPAREN",
            TokenType::Equals => "EQUALS",
            TokenType::Eof => "EOF",
            _ => panic!("unexpected token type"),
        }).collect();
        assert_eq!(
            types,
            &["IDENT", "ARROW", "IDENT", "LPAREN", "IDENT", "EQUALS", "NUMBER", "RPAREN", "ARROW", "IDENT", "EOF"]
        );
    }
}
