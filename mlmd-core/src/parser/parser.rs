use crate::ast::nodes::*;
use crate::parser::tokenizer::{Token, TokenType};

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub loc: SourceLoc,
}

impl ParseError {
    pub fn new(message: impl Into<String>, loc: SourceLoc) -> Self {
        Self {
            message: message.into(),
            loc,
        }
    }
}

// ---------------------------------------------------------------------------
// ParseResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub nodes: Vec<ASTNode>,
    pub errors: Vec<ParseError>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    errors: Vec<ParseError>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    fn peek(&self) -> &'a Token {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos]
        } else {
            // Should not happen with EOF sentinel, but guard anyway
            &self.tokens[self.tokens.len() - 1]
        }
    }

    fn advance(&mut self) -> &'a Token {
        let t = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, token_type: TokenType) -> Option<&'a Token> {
        if self.peek().token_type == token_type {
            Some(self.advance())
        } else {
            None
        }
    }

    fn skip_to_next_line(&mut self) {
        while self.peek().token_type != TokenType::Newline
            && self.peek().token_type != TokenType::Eof
        {
            self.advance();
        }
        self.eat(TokenType::Newline);
    }

    fn parse_number(&self, tok: &Token) -> f64 {
        tok.value.parse::<f64>().unwrap_or(0.0)
    }

    fn parse_param_value(&mut self) -> Option<ParamValue> {
        let t = self.peek();

        // Shape: (1,28,28)
        if t.token_type == TokenType::LParen {
            let loc = t.loc.clone();
            self.advance(); // (
            let mut dims: Vec<usize> = Vec::new();
            while self.peek().token_type != TokenType::RParen
                && self.peek().token_type != TokenType::Eof
                && self.peek().token_type != TokenType::Newline
            {
                let num_tok = self.eat(TokenType::Number);
                match num_tok {
                    Some(n) => {
                        let v = self.parse_number(n) as usize;
                        dims.push(v);
                    }
                    None => {
                        self.errors.push(ParseError::new(
                            format!("Expected number in shape, got {:?}", self.peek().token_type),
                            self.peek().loc.clone(),
                        ));
                        return None;
                    }
                }
                if self.eat(TokenType::Comma).is_none() {
                    break;
                }
            }
            if self.eat(TokenType::RParen).is_none() {
                self.errors.push(ParseError::new(
                    "Expected ) to close shape",
                    self.peek().loc.clone(),
                ));
                return None;
            }
            return Some(ParamValue::Shape(Box::new(ShapeVal::new(dims, loc))));
        }

        // List: [64,128,256]
        if t.token_type == TokenType::LBracket {
            let loc = t.loc.clone();
            self.advance(); // [
            let mut items: Vec<ParamValue> = Vec::new();
            while self.peek().token_type != TokenType::RBracket
                && self.peek().token_type != TokenType::Eof
                && self.peek().token_type != TokenType::Newline
            {
                let val = self.parse_param_value();
                match val {
                    Some(v) => items.push(v),
                    None => return None,
                }
                if self.eat(TokenType::Comma).is_none() {
                    break;
                }
            }
            if self.eat(TokenType::RBracket).is_none() {
                self.errors.push(ParseError::new(
                    "Expected ] to close list",
                    self.peek().loc.clone(),
                ));
                return None;
            }
            return Some(ParamValue::List(Box::new(ListVal::new(items, loc))));
        }

        // Number
        if t.token_type == TokenType::Number {
            self.advance();
            return Some(ParamValue::Number(Box::new(NumberVal::new(
                self.parse_number(t),
                t.loc.clone(),
            ))));
        }

        // String
        if t.token_type == TokenType::String {
            self.advance();
            return Some(ParamValue::String(Box::new(StringVal::new(
                t.value.clone(),
                t.loc.clone(),
            ))));
        }

        // Bool
        if t.token_type == TokenType::Bool {
            self.advance();
            return Some(ParamValue::Bool(Box::new(BoolVal::new(
                t.value == "true",
                t.loc.clone(),
            ))));
        }

        // Bareword (IDENT used as a value)
        if t.token_type == TokenType::Ident {
            self.advance();
            return Some(ParamValue::Bareword(Box::new(BarewordVal::new(
                t.value.clone(),
                t.loc.clone(),
            ))));
        }

        self.errors.push(ParseError::new(
            format!(
                "Unexpected token {:?} ({}) in param value",
                t.token_type, t.value
            ),
            t.loc.clone(),
        ));
        None
    }

    fn parse_param(&mut self) -> Option<Param> {
        let name_tok = self.eat(TokenType::Ident);
        let name_tok = match name_tok {
            Some(t) => t,
            None => {
                self.errors.push(ParseError::new(
                    format!("Expected param name, got {:?}", self.peek().token_type),
                    self.peek().loc.clone(),
                ));
                return None;
            }
        };

        if self.eat(TokenType::Equals).is_none() {
            self.errors.push(ParseError::new(
                format!(
                    "Expected = after param name \"{}\"",
                    name_tok.value
                ),
                self.peek().loc.clone(),
            ));
            return None;
        }

        let value = self.parse_param_value();
        let value = match value {
            Some(v) => v,
            None => return None,
        };

        Some(Param {
            name: name_tok.value.clone(),
            value,
            loc: name_tok.loc.clone(),
        })
    }

    fn parse_block_decl(&mut self, ident_tok: &Token) -> Option<BlockDecl> {
        let mut params: Vec<Param> = Vec::new();

        if self.eat(TokenType::LParen).is_some() {
            // parse params — may span multiple lines (continuation with trailing comma)
            loop {
                // skip newlines inside param list (multi-line params)
                while self.peek().token_type == TokenType::Newline {
                    self.advance();
                }
                if self.peek().token_type == TokenType::RParen
                    || self.peek().token_type == TokenType::Eof
                {
                    break;
                }
                let param = self.parse_param();
                match param {
                    Some(p) => params.push(p),
                    None => return None,
                }
                if self.eat(TokenType::Comma).is_none() {
                    break;
                }
            }
            if self.eat(TokenType::RParen).is_none() {
                self.errors.push(ParseError::new(
                    "Expected ) to close param list",
                    self.peek().loc.clone(),
                ));
                return None;
            }
        }

        Some(BlockDecl::new(
            ident_tok.value.clone(),
            params,
            ident_tok.loc.clone(),
        ))
    }

    // Parse a bracket list: [ name1, name2, ... ] — returns names or None on error
    // Caller already consumed LBRACKET.
    fn parse_bracket_names(&mut self, _loc: &SourceLoc) -> Option<Vec<String>> {
        let mut names: Vec<String> = Vec::new();
        while self.peek().token_type != TokenType::RBracket
            && self.peek().token_type != TokenType::Eof
            && self.peek().token_type != TokenType::Newline
        {
            let name_tok = self.eat(TokenType::Ident);
            match name_tok {
                Some(t) => names.push(t.value.clone()),
                None => {
                    self.errors.push(ParseError::new(
                        format!(
                            "Expected identifier in bracket list, got {:?}",
                            self.peek().token_type
                        ),
                        self.peek().loc.clone(),
                    ));
                    return None;
                }
            }
            if self.eat(TokenType::Comma).is_none() {
                break;
            }
        }
        if self.eat(TokenType::RBracket).is_none() {
            self.errors.push(ParseError::new(
                "Expected ] to close tensor name list",
                self.peek().loc.clone(),
            ));
            return None;
        }
        Some(names)
    }

    // After parsing a block, handle zero or more `-> ...` continuations.
    // `pending_sources` is set when this chain started from a TensorJoin ([a,b] ->).
    // `out` receives emitted nodes (either top-level or a group body).
    fn parse_chain_tail(
        &mut self,
        first_block: BlockDecl,
        pending_sources: Option<Vec<String>>,
        out: &mut Vec<ASTNode>,
    ) {
        let mut current_block = first_block;
        let mut sources: Option<Vec<String>> = pending_sources;

        loop {
            // Support multi-line chains: after a NEWLINE, if next token is ARROW, continue
            if self.peek().token_type == TokenType::Newline {
                // Look ahead past the newline to see if next meaningful token is ARROW
                let mut lookahead = self.pos + 1;
                while lookahead < self.tokens.len()
                    && self.tokens[lookahead].token_type == TokenType::Newline
                {
                    lookahead += 1;
                }
                if lookahead >= self.tokens.len()
                    || self.tokens[lookahead].token_type != TokenType::Arrow
                {
                    break;
                }
                // consume newlines up to the arrow
                while self.peek().token_type == TokenType::Newline {
                    self.advance();
                }
            }

            if self.peek().token_type != TokenType::Arrow {
                break;
            }
            self.advance(); // consume ->

            let next = self.peek().clone();

            if next.token_type == TokenType::LBracket {
                // Fork: Block -> [a, b]  OR  Block -> [a]
                let loc = next.loc.clone();
                self.advance(); // [
                let names = self.parse_bracket_names(&loc);
                let names = match names {
                    Some(n) => n,
                    None => {
                        self.skip_to_next_line();
                        return;
                    }
                };

                // Emit the current block (possibly as part of a join)
                if let Some(ref srcs) = sources {
                    let block_loc = current_block.loc.clone();
                    out.push(ASTNode::TensorJoin(Box::new(TensorJoin::new(
                        srcs.clone(),
                        current_block,
                        block_loc,
                    ))));
                } else {
                    out.push(ASTNode::Block(Box::new(current_block)));
                }

                // Emit the TensorName
                out.push(ASTNode::TensorName(Box::new(TensorName::new(names, loc))));

                // After a fork we expect end of line (no further chaining from names)
                self.eat(TokenType::Newline);
                return;
            }

            if next.token_type == TokenType::Ident {
                // Another block in the chain — advance past the ident token
                self.advance();
                let next_block = self.parse_block_decl(&next);
                let next_block = match next_block {
                    Some(b) => b,
                    None => {
                        self.skip_to_next_line();
                        return;
                    }
                };

                // Emit the current block (possibly as join)
                if let Some(ref srcs) = sources {
                    let block_loc = current_block.loc.clone();
                    out.push(ASTNode::TensorJoin(Box::new(TensorJoin::new(
                        srcs.clone(),
                        current_block,
                        block_loc,
                    ))));
                    sources = None;
                } else {
                    out.push(ASTNode::Block(Box::new(current_block)));
                }

                current_block = next_block;
                continue;
            }

            // Unexpected token after ->
            self.errors.push(ParseError::new(
                format!("Expected block or [ after ->, got {:?}", next.token_type),
                next.loc.clone(),
            ));
            self.skip_to_next_line();
            return;
        }

        // No more arrows — emit the final block
        if let Some(ref srcs) = sources {
            let block_loc = current_block.loc.clone();
            out.push(ASTNode::TensorJoin(Box::new(TensorJoin::new(
                srcs.clone(),
                current_block,
                block_loc,
            ))));
        } else {
            out.push(ASTNode::Block(Box::new(current_block)));
        }
        self.eat(TokenType::Newline);
    }

    // Parse a group declaration: GROUP_OPEN already consumed.
    fn parse_group_decl(&mut self, loc: &SourceLoc) -> Option<GroupDecl> {
        let mut path: Vec<String> = Vec::new();

        // Expect: IDENT (GT IDENT)* GROUP_CLOSE
        let first = self.eat(TokenType::Ident);
        let first = match first {
            Some(t) => t,
            None => {
                self.errors.push(ParseError::new(
                    "Expected group name after [[",
                    self.peek().loc.clone(),
                ));
                self.skip_to_next_line();
                return None;
            }
        };
        path.push(first.value.clone());

        while self.peek().token_type == TokenType::Gt {
            self.advance(); // consume >
            let seg = self.eat(TokenType::Ident);
            let seg = match seg {
                Some(t) => t,
                None => {
                    self.errors.push(ParseError::new(
                        "Expected group name segment after >",
                        self.peek().loc.clone(),
                    ));
                    self.skip_to_next_line();
                    return None;
                }
            };
            path.push(seg.value.clone());
        }

        if self.eat(TokenType::GroupClose).is_none() {
            self.errors.push(ParseError::new(
                "Expected ]] to close group declaration",
                self.peek().loc.clone(),
            ));
            self.skip_to_next_line();
            return None;
        }

        // optional trailing comment / newline
        self.eat(TokenType::Comment);
        self.eat(TokenType::Newline);

        Some(GroupDecl::new(path, Vec::new(), loc.clone()))
    }

    fn parse_line(&mut self, out: &mut Vec<ASTNode>) {
        // skip blank lines
        while self.eat(TokenType::Newline).is_some() {}

        let t = self.peek().clone();

        if t.token_type == TokenType::Eof {
            return;
        }

        if t.token_type == TokenType::Comment {
            self.advance();
            out.push(ASTNode::Comment(Box::new(Comment::new(
                t.value.clone(),
                t.loc.clone(),
            ))));
            self.eat(TokenType::Newline);
            return;
        }

        // Group declaration: [[ ... ]]
        if t.token_type == TokenType::GroupOpen {
            self.advance();
            let group = self.parse_group_decl(&t.loc);
            let mut group = match group {
                Some(g) => g,
                None => return,
            };

            // Collect body lines until next GROUP_OPEN or EOF
            while self.peek().token_type != TokenType::Eof
                && self.peek().token_type != TokenType::GroupOpen
            {
                // skip blank lines
                while self.eat(TokenType::Newline).is_some() {}
                if self.peek().token_type == TokenType::Eof
                    || self.peek().token_type == TokenType::GroupOpen
                {
                    break;
                }
                self.parse_line(&mut group.body);
            }

            out.push(ASTNode::Group(Box::new(group)));
            return;
        }

        // Join: [a, b] -> Block
        if t.token_type == TokenType::LBracket {
            let loc = t.loc.clone();
            self.advance(); // [
            let names = self.parse_bracket_names(&loc);
            let names = match names {
                Some(n) => n,
                None => {
                    self.skip_to_next_line();
                    return;
                }
            };

            if self.eat(TokenType::Arrow).is_none() {
                self.errors.push(ParseError::new(
                    "Expected -> after tensor list",
                    self.peek().loc.clone(),
                ));
                self.skip_to_next_line();
                return;
            }

            let ident_tok = self.eat(TokenType::Ident);
            let ident_tok = match ident_tok {
                Some(t) => t,
                None => {
                    self.errors.push(ParseError::new(
                        format!(
                            "Expected block name after ->, got {:?}",
                            self.peek().token_type
                        ),
                        self.peek().loc.clone(),
                    ));
                    self.skip_to_next_line();
                    return;
                }
            };

            let block = self.parse_block_decl(ident_tok);
            let block = match block {
                Some(b) => b,
                None => {
                    self.skip_to_next_line();
                    return;
                }
            };

            self.parse_chain_tail(block, Some(names), out);
            return;
        }

        if t.token_type == TokenType::Ident {
            self.advance();
            let block_decl = self.parse_block_decl(&t);
            let block_decl = match block_decl {
                Some(b) => b,
                None => {
                    self.skip_to_next_line();
                    return;
                }
            };

            // Check for chain
            if self.peek().token_type == TokenType::Arrow {
                self.parse_chain_tail(block_decl, None, out);
                return;
            }

            // Check for multi-line chain (NEWLINE followed by ARROW)
            if self.peek().token_type == TokenType::Newline {
                let mut lookahead = self.pos + 1;
                while lookahead < self.tokens.len()
                    && self.tokens[lookahead].token_type == TokenType::Newline
                {
                    lookahead += 1;
                }
                if lookahead < self.tokens.len()
                    && self.tokens[lookahead].token_type == TokenType::Arrow
                {
                    self.parse_chain_tail(block_decl, None, out);
                    return;
                }
            }

            out.push(ASTNode::Block(Box::new(block_decl)));
            self.eat(TokenType::Newline);
            return;
        }

        // Unknown line start — skip
        self.errors.push(ParseError::new(
            format!(
                "Unexpected token {:?} at start of line",
                t.token_type
            ),
            t.loc.clone(),
        ));
        self.skip_to_next_line();
    }

    fn parse_all(mut self) -> ParseResult {
        let mut nodes: Vec<ASTNode> = Vec::new();

        while self.peek().token_type != TokenType::Eof {
            self.parse_line(&mut nodes);
        }

        ParseResult {
            nodes,
            errors: self.errors,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse(tokens: &[Token]) -> ParseResult {
    let parser = Parser::new(tokens);
    parser.parse_all()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tokenizer::tokenize;

    fn parse_source(src: &str) -> ParseResult {
        let tokens = tokenize(src);
        parse(&tokens)
    }

    // ---- block declarations -----------------------------------------------

    #[test]
    fn test_bare_block_no_params() {
        let result = parse_source("ReLU");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.block_type, "ReLU");
                assert_eq!(b.params.len(), 0);
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_number_params() {
        let result = parse_source("Conv2d(kernel=5, filters=6)");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.block_type, "Conv2d");
                assert_eq!(b.params.len(), 2);
                assert_eq!(b.params[0].name, "kernel");
                match &b.params[0].value {
                    ParamValue::Number(v) => assert_eq!(v.value, 5.0),
                    _ => panic!("expected number"),
                }
                assert_eq!(b.params[1].name, "filters");
                match &b.params[1].value {
                    ParamValue::Number(v) => assert_eq!(v.value, 6.0),
                    _ => panic!("expected number"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_float_param() {
        let result = parse_source("Dropout(rate=0.5)");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params[0].name, "rate");
                match &b.params[0].value {
                    ParamValue::Number(v) => assert_eq!(v.value, 0.5),
                    _ => panic!("expected number"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_negative_number_param() {
        let result = parse_source("Foo(x=-3)");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params[0].name, "x");
                match &b.params[0].value {
                    ParamValue::Number(v) => assert_eq!(v.value, -3.0),
                    _ => panic!("expected number"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_string_param() {
        let result = parse_source(r#"Layer(name="encoder")"#);
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params[0].name, "name");
                match &b.params[0].value {
                    ParamValue::String(v) => assert_eq!(v.value, "encoder"),
                    _ => panic!("expected string"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_bool_params() {
        let result = parse_source("BN(affine=true, track=false)");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params[0].name, "affine");
                match &b.params[0].value {
                    ParamValue::Bool(v) => assert_eq!(v.value, true),
                    _ => panic!("expected bool"),
                }
                assert_eq!(b.params[1].name, "track");
                match &b.params[1].value {
                    ParamValue::Bool(v) => assert_eq!(v.value, false),
                    _ => panic!("expected bool"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_bareword_param() {
        let result = parse_source("Conv2d(padding=same)");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params[0].name, "padding");
                match &b.params[0].value {
                    ParamValue::Bareword(v) => assert_eq!(v.value, "same"),
                    _ => panic!("expected bareword"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_shape_param() {
        let result = parse_source("Input(shape=(1,28,28))");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.block_type, "Input");
                assert_eq!(b.params[0].name, "shape");
                match &b.params[0].value {
                    ParamValue::Shape(v) => {
                        assert_eq!(v.dims, vec![1, 28, 28]);
                    }
                    _ => panic!("expected shape"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_block_with_list_param() {
        let result = parse_source("MultiScale(filters=[64,128,256])");
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params[0].name, "filters");
                match &b.params[0].value {
                    ParamValue::List(l) => {
                        assert_eq!(l.items.len(), 3);
                        match &l.items[0] {
                            ParamValue::Number(v) => assert_eq!(v.value, 64.0),
                            _ => panic!("expected number"),
                        }
                        match &l.items[1] {
                            ParamValue::Number(v) => assert_eq!(v.value, 128.0),
                            _ => panic!("expected number"),
                        }
                        match &l.items[2] {
                            ParamValue::Number(v) => assert_eq!(v.value, 256.0),
                            _ => panic!("expected number"),
                        }
                    }
                    _ => panic!("expected list"),
                }
            }
            _ => panic!("expected Block node"),
        }
    }

    #[test]
    fn test_multiple_blocks_on_separate_lines() {
        let result = parse_source("Conv2d(filters=32)\nReLU\nMaxPool(size=2)");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 3);
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[1] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "ReLU"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[2] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "MaxPool"),
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn test_comments() {
        let result = parse_source("# a comment\nReLU");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 2);
        match &result.nodes[0] {
            ASTNode::Comment(c) => {
                assert_eq!(c.text, "a comment");
            }
            _ => panic!("expected Comment"),
        }
        match &result.nodes[1] {
            ASTNode::Block(_) => {}
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn test_skips_faulty_lines_and_continues() {
        let result = parse_source("Conv2d(filters=32)\n=bad\nReLU");
        assert!(result.errors.len() > 0);
        let blocks: Vec<&str> = result
            .nodes
            .iter()
            .filter_map(|n| match n {
                ASTNode::Block(b) => Some(b.block_type.as_str()),
                _ => None,
            })
            .collect();
        assert!(blocks.contains(&"Conv2d"));
        assert!(blocks.contains(&"ReLU"));
    }

    #[test]
    fn test_recovers_after_missing_closing_paren() {
        let result = parse_source("Bad(x=1\nReLU");
        assert!(result.errors.len() > 0);
        let blocks: Vec<&str> = result
            .nodes
            .iter()
            .filter_map(|n| match n {
                ASTNode::Block(b) => Some(b.block_type.as_str()),
                _ => None,
            })
            .collect();
        assert!(blocks.contains(&"ReLU"));
    }

    #[test]
    fn test_records_correct_loc_for_block() {
        let result = parse_source("ReLU");
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.loc.line, 1);
                assert_eq!(b.loc.col, 1);
            }
            _ => panic!("expected Block"),
        }
    }

    // ---- arrow chains -----------------------------------------------------

    #[test]
    fn test_simple_oneliner_chain() {
        let result = parse_source("Input -> ReLU -> Output");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 3);
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Input"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[1] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "ReLU"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[2] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Output"),
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn test_block_to_single_tensor_name() {
        let result = parse_source("Conv2d(kernel=3) -> [skip]");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 2);
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[1] {
            ASTNode::TensorName(tn) => {
                assert_eq!(tn.names, vec!["skip"]);
            }
            _ => panic!("expected TensorName"),
        }
    }

    #[test]
    fn test_block_to_fork() {
        let result = parse_source("ReLU -> [skip, main]");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 2);
        match &result.nodes[1] {
            ASTNode::TensorName(tn) => {
                assert_eq!(tn.names, vec!["skip", "main"]);
            }
            _ => panic!("expected TensorName"),
        }
    }

    #[test]
    fn test_join_a_b_to_block() {
        let result = parse_source("[skip, main] -> Add");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::TensorJoin(tj) => {
                assert_eq!(tj.sources, vec!["skip", "main"]);
                assert_eq!(tj.target.block_type, "Add");
            }
            _ => panic!("expected TensorJoin"),
        }
    }

    #[test]
    fn test_long_chain_with_tensor_name() {
        let result = parse_source("Input -> Conv2d(kernel=1) -> ReLU -> [skip]");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 4);
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Input"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[1] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[2] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "ReLU"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[3] {
            ASTNode::TensorName(tn) => assert_eq!(tn.names, vec!["skip"]),
            _ => panic!("expected TensorName"),
        }
    }

    #[test]
    fn test_standalone_block_no_chain() {
        let result = parse_source("Conv2d(kernel=3, stride=1)");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
    }

    // ---- groups -----------------------------------------------------------

    #[test]
    fn test_single_level_group_no_body() {
        let result = parse_source("[[ Encoder ]]");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Group(g) => {
                assert_eq!(g.path, vec!["Encoder"]);
                assert_eq!(g.body.len(), 0);
            }
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_nested_group_path() {
        let result = parse_source("[[ Encoder > Block1 ]]");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Group(g) => {
                assert_eq!(g.path, vec!["Encoder", "Block1"]);
            }
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_three_level_group_path() {
        let result = parse_source("[[ A > B > C ]]");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Group(g) => {
                assert_eq!(g.path, vec!["A", "B", "C"]);
            }
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_group_body_contains_block_nodes() {
        let result = parse_source("[[ Encoder ]]\nConv2d (kernel_size=3)\nReLU\n");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Group(g) => {
                assert_eq!(g.body.len(), 2);
                match &g.body[0] {
                    ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
                    _ => panic!("expected Block in group body"),
                }
                match &g.body[1] {
                    ASTNode::Block(b) => assert_eq!(b.block_type, "ReLU"),
                    _ => panic!("expected Block in group body"),
                }
            }
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_group_body_ends_at_next_group() {
        let src = "[[ GroupA ]]\nConv2d (kernel_size=3)\n[[ GroupB ]]\nReLU\n";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 2);
        match &result.nodes[0] {
            ASTNode::Group(a) => {
                assert_eq!(a.path, vec!["GroupA"]);
                assert_eq!(a.body.len(), 1);
            }
            _ => panic!("expected Group"),
        }
        match &result.nodes[1] {
            ASTNode::Group(b) => {
                assert_eq!(b.path, vec!["GroupB"]);
                assert_eq!(b.body.len(), 1);
            }
            _ => panic!("expected Group"),
        }
    }

    #[test]
    fn test_comment_line() {
        let result = parse_source("# this is a comment");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 1);
        match &result.nodes[0] {
            ASTNode::Comment(c) => {
                assert_eq!(c.text, "this is a comment");
            }
            _ => panic!("expected Comment"),
        }
    }

    #[test]
    fn test_preserves_comments_in_group_body() {
        let src = "[[ Section ]]\n# a comment\nReLU\n";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Group(g) => {
                assert_eq!(g.body.len(), 2);
                match &g.body[0] {
                    ASTNode::Comment(c) => assert_eq!(c.text, "a comment"),
                    _ => panic!("expected Comment in group body"),
                }
                match &g.body[1] {
                    ASTNode::Block(_) => {}
                    _ => panic!("expected Block in group body"),
                }
            }
            _ => panic!("expected Group"),
        }
    }

    // ---- multi-line chains ------------------------------------------------

    #[test]
    fn test_lenet_style_multi_line_chain() {
        let src = "Input (shape=(1,28,28))\n    -> Conv2d (kernel_size=5, filters=6)\n    -> Tanh ()";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 3);
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Input"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[1] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[2] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Tanh"),
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn test_multi_line_chain_ending_in_fork() {
        let src = "Input (shape=(256,56,56))\n    -> [main, skip]";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 2);
        match &result.nodes[1] {
            ASTNode::TensorName(tn) => {
                assert_eq!(tn.names, vec!["main", "skip"]);
            }
            _ => panic!("expected TensorName"),
        }
    }

    // ---- Lenet-5 example --------------------------------------------------

    #[test]
    fn test_lenet5_without_error() {
        let src = "Input (shape=(1,28,28))
    -> Conv2d (kernel_size=5, filters=6)    -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)
    -> Conv2d (kernel_size=5, filters=16)   -> Tanh ()
    -> AvgPool (kernel_size=2, stride=2)
    -> Flatten () -> Linear (out_features=120) -> Tanh ()
    -> Linear (out_features=84) -> Tanh ()
    -> Linear (out_features=10) -> Softmax ()";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
    }

    // ---- ResNet bottleneck example ----------------------------------------

    #[test]
    fn test_resnet_bottleneck_without_error() {
        let src = "[[ ResNet50 > Bottleneck ]]

Input (shape=(256, 56, 56)) -> [main, skip]

main -> Conv2d (kernel_size=1, filters=64)  -> BatchNorm () -> ReLU ()
    -> Conv2d (kernel_size=3, filters=64, padding=same) -> BatchNorm () -> ReLU ()
    -> Conv2d (kernel_size=1, filters=256)  -> BatchNorm () -> [main_out]

skip -> Conv2d (kernel_size=1, filters=256) -> BatchNorm () -> [skip_out]

[main_out, skip_out] -> Add () -> ReLU ()";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
    }

    // ---- Attention head example -------------------------------------------

    #[test]
    fn test_attention_head_without_error() {
        let src = "Input (shape=(512, 64)) -> [query]
Input (shape=(512, 64)) -> [key]
Input (shape=(512, 64)) -> [value]

query -> Linear (out_features=64, bias=false) -> [q_proj]
key   -> Linear (out_features=64, bias=false) -> [k_proj]
value -> Linear (out_features=64, bias=false) -> [v_proj]

[q_proj, k_proj] -> MatMul () -> Mul (scalar=0.125) -> Softmax (dim=-1) -> [attn]

[attn, v_proj] -> MatMul () -> Linear (out_features=64)";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
    }

    // ---- ResNet bottleneck pattern (fork + join) --------------------------

    #[test]
    fn test_resnet_bottleneck_pattern() {
        let src = "Input -> Conv2d(kernel=1) -> ReLU -> [skip]
Input -> Conv2d(kernel=3) -> ReLU -> [main]
[skip, main] -> Add -> Output";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 10);

        // Line 1: Input, Conv2d, ReLU, TensorName[skip]
        match &result.nodes[0] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Input"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[1] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[2] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "ReLU"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[3] {
            ASTNode::TensorName(tn) => assert_eq!(tn.names, vec!["skip"]),
            _ => panic!("expected TensorName"),
        }

        // Line 2: Input, Conv2d, ReLU, TensorName[main]
        match &result.nodes[4] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Input"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[5] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Conv2d"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[6] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "ReLU"),
            _ => panic!("expected Block"),
        }
        match &result.nodes[7] {
            ASTNode::TensorName(tn) => assert_eq!(tn.names, vec!["main"]),
            _ => panic!("expected TensorName"),
        }

        // Line 3: TensorJoin([skip,main] -> Add), Output
        match &result.nodes[8] {
            ASTNode::TensorJoin(tj) => {
                assert_eq!(tj.sources, vec!["skip", "main"]);
                assert_eq!(tj.target.block_type, "Add");
            }
            _ => panic!("expected TensorJoin"),
        }
        match &result.nodes[9] {
            ASTNode::Block(b) => assert_eq!(b.block_type, "Output"),
            _ => panic!("expected Block"),
        }
    }

    // ---- empty input ------------------------------------------------------

    #[test]
    fn test_empty_input() {
        let result = parse_source("");
        assert_eq!(result.errors.len(), 0);
        assert_eq!(result.nodes.len(), 0);
    }

    // ---- error cases ------------------------------------------------------

    #[test]
    fn test_unexpected_token_at_start_of_line() {
        let result = parse_source("=bad");
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_malformed_param_missing_equals() {
        let result = parse_source("Block(x 5)");
        // x is parsed as param name, then 5 without = is an error
        // actually, x would be consumed as param name, no = follows, so parse_param errors
        assert!(result.errors.len() > 0);
    }

    #[test]
    fn test_shape_with_non_number() {
        let result = parse_source("Block(s=(x,))");
        assert!(result.errors.len() > 0);
    }

    #[test]
    fn test_list_unclosed_bracket() {
        let result = parse_source("MultiScale(filters=[64)");
        assert!(result.errors.len() > 0);
    }

    // ---- multi-line param lists -------------------------------------------

    #[test]
    fn test_multi_line_params() {
        let src = "Block(\n  x=1,\n  y=2)";
        let result = parse_source(src);
        assert_eq!(result.errors.len(), 0);
        match &result.nodes[0] {
            ASTNode::Block(b) => {
                assert_eq!(b.params.len(), 2);
                assert_eq!(b.params[0].name, "x");
                assert_eq!(b.params[1].name, "y");
            }
            _ => panic!("expected Block"),
        }
    }
}
