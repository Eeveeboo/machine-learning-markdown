// ---------------------------------------------------------------------------
// LSP completions — context-aware completion items for MLMD
// ---------------------------------------------------------------------------

use tower_lsp::lsp_types::{CompletionItem, CompletionItemKind};

use crate::block::registry::{all_block_names, lookup_block};

// ---------------------------------------------------------------------------
// Tensor name extraction
// ---------------------------------------------------------------------------

/// Extract all tensor names declared via `-> [tensorName]` in the document source.
fn get_named_tensors(source: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut search_pos = 0;
    let bytes = source.as_bytes();
    while search_pos < bytes.len() {
        // Find "->"
        let arrow_start = memchr(b'-', &bytes[search_pos..]);
        let arrow_start = match arrow_start {
            Some(pos) => search_pos + pos,
            None => break,
        };
        if arrow_start + 1 < bytes.len() && bytes[arrow_start + 1] == b'>' {
            // Found "->", look for '[' after it
            let after_arrow = arrow_start + 2;
            let bracket_start = memchr(b'[', &bytes[after_arrow..]);
            let bracket_start = match bracket_start {
                Some(pos) => after_arrow + pos,
                None => {
                    search_pos = after_arrow;
                    continue;
                }
            };
            let bracket_end = memchr(b']', &bytes[bracket_start + 1..]);
            let bracket_end = match bracket_end {
                Some(pos) => bracket_start + 1 + pos,
                None => {
                    search_pos = bracket_start + 1;
                    continue;
                }
            };
            // Extract content between brackets
            let content = &source[bracket_start + 1..bracket_end];
            for part in content.split(',') {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    names.push(trimmed.to_string());
                }
            }
            search_pos = bracket_end + 1;
        } else {
            search_pos = arrow_start + 1;
        }
    }
    names
}

// Simple memchr implementation since we don't want extra deps
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

// ---------------------------------------------------------------------------
// Context detection
// ---------------------------------------------------------------------------

/// Determine what kind of completion context the cursor is in.
enum CompletionContext {
    /// Cursor is inside `[...]` brackets → suggest tensor names
    InsideBrackets,
    /// Cursor is inside `(...)` parens → suggest param names for the current block
    InsideParens { block_type: String },
    /// Cursor is at line start or after `->` → suggest block type names
    BlockType,
    /// No completion context detected
    None,
}

/// Analyze the document and cursor position to determine completion context.
fn detect_context(source: &str, line: u32, character: u32) -> CompletionContext {
    let lines: Vec<&str> = source.lines().collect();
    let line_idx = line as usize;

    // Get the full text up to the cursor position
    let up_to_cursor: String = source
        .lines()
        .enumerate()
        .take_while(|(i, _)| *i < line_idx)
        .map(|(_, l)| format!("{}\n", l))
        .chain(std::iter::once({
            let current_line = lines.get(line_idx).unwrap_or(&"");
            current_line.chars().take(character as usize).collect::<String>()
        }))
        .collect();

    // Check if we're inside `[...]`
    let last_open_bracket = up_to_cursor.rfind('[');
    let last_close_bracket = up_to_cursor.rfind(']');
    if let Some(ob) = last_open_bracket {
        if last_close_bracket.map_or(true, |cb| ob > cb) {
            return CompletionContext::InsideBrackets;
        }
    }

    // Check if we're inside `(...)`  — look for param context
    let last_open_paren = up_to_cursor.rfind('(');
    let last_close_paren = up_to_cursor.rfind(')');
    if let Some(op) = last_open_paren {
        if last_close_paren.map_or(true, |cp| op > cp) {
            // Try to find the block type name before the paren
            let before_paren = &up_to_cursor[..op];
            if let Some(block_type) = extract_block_type_before_paren(before_paren) {
                return CompletionContext::InsideParens { block_type };
            }
        }
    }

    // Check if at line start or after `->`
    let current_line = lines.get(line_idx).unwrap_or(&"");
    let before_cursor = &current_line[..character as usize];

    let trimmed = before_cursor.trim_start();
    if trimmed.is_empty() || before_cursor.ends_with("->") || before_cursor.contains("->") {
        let after_arrow = before_cursor.rfind("->").map(|i| &before_cursor[i + 2..]);
        if let Some(after) = after_arrow {
            if after.chars().all(|c| c.is_whitespace() || c.is_alphanumeric() || c == '_') {
                return CompletionContext::BlockType;
            }
        } else if trimmed.is_empty() {
            return CompletionContext::BlockType;
        }
    }

    CompletionContext::None
}

/// Given text before `(`, try to extract the block type name.
fn extract_block_type_before_paren(before_paren: &str) -> Option<String> {
    // Look for the last word before '(' that looks like a block type
    let trimmed = before_paren.trim_end();
    // Check for patterns like "BlockType (" or "-> BlockType ("
    if let Some(arrow_pos) = trimmed.rfind("->") {
        let after_arrow = trimmed[arrow_pos + 2..].trim();
        if !after_arrow.is_empty() && is_valid_ident(after_arrow) {
            return Some(after_arrow.to_string());
        }
    }
    // Fallback: take the last word
    let last_word = trimmed.rsplit(|c: char| c.is_whitespace()).next()?;
    if is_valid_ident(last_word) && !last_word.is_empty() {
        Some(last_word.to_string())
    } else {
        None
    }
}

fn is_valid_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
        && s.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_')
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Get completion items at the given cursor position in the document.
pub fn get_completions(doc_source: &str, line: u32, character: u32) -> Vec<CompletionItem> {
    let context = detect_context(doc_source, line, character);

    match context {
        CompletionContext::InsideBrackets => {
            // Suggest tensor names from the document
            let tensors = get_named_tensors(doc_source);
            tensors
                .into_iter()
                .map(|name| CompletionItem {
                    label: name,
                    kind: Some(CompletionItemKind::VARIABLE),
                    ..Default::default()
                })
                .collect()
        }

        CompletionContext::InsideParens { block_type } => {
            // Suggest param names for the given block type
            if let Some(def) = lookup_block(&block_type) {
                let params = def.params();
                params
                    .iter()
                    .map(|p| CompletionItem {
                        label: p.name.clone(),
                        kind: Some(CompletionItemKind::PROPERTY),
                        insert_text: Some(format!("{}=", p.name)),
                        ..Default::default()
                    })
                    .collect()
            } else {
                vec![]
            }
        }

        CompletionContext::BlockType => {
            // Suggest all registered block type names
            all_block_names()
                .into_iter()
                .map(|name| CompletionItem {
                    label: name,
                    kind: Some(CompletionItemKind::CLASS),
                    ..Default::default()
                })
                .collect()
        }

        CompletionContext::None => vec![],
    }
}
