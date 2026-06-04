// ---------------------------------------------------------------------------
// LSP references — find all usages of a tensor name across the document
// ---------------------------------------------------------------------------

use tower_lsp::lsp_types::{Location, Position, Range, Url};

use crate::lsp::word_at_position;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find all references to a tensor name across the document.
///
/// Searches all `[...]` bracket groups (declarations and join patterns)
/// for occurrences of the tensor name under the cursor.
pub fn get_references(doc_source: &str, uri: &Url, position: &Position) -> Vec<Location> {
    let lines: Vec<&str> = doc_source.lines().collect();
    let line = match lines.get(position.line as usize) {
        Some(l) => l,
        None => return vec![],
    };
    let tensor_name = match word_at_position(line, position.character) {
        Some(n) if !n.is_empty() => n,
        _ => return vec![],
    };

    let mut locations: Vec<Location> = Vec::new();

    for (line_idx, line_text) in lines.iter().enumerate() {
        let mut search_pos = 0;
        // Find all `[...]` groups on this line
        while let Some(open_pos) = line_text[search_pos..].find('[') {
            let abs_open = search_pos + open_pos;
            let content_start = abs_open + 1;

            // Find closing bracket
            if let Some(close_pos) = line_text[content_start..].find(']') {
                let abs_close = content_start + close_pos;
                let content = &line_text[content_start..abs_close];

                // Check if our tensor name is in this bracket content
                for part in content.split(',') {
                    let trimmed = part.trim();
                    if trimmed == tensor_name {
                        // Find the exact position of the name within the bracket content
                        if let Some(name_offset) = content.find(&tensor_name) {
                            let col = content_start + name_offset;
                            locations.push(Location {
                                uri: uri.clone(),
                                range: Range {
                                    start: Position {
                                        line: line_idx as u32,
                                        character: col as u32,
                                    },
                                    end: Position {
                                        line: line_idx as u32,
                                        character: (col + tensor_name.len()) as u32,
                                    },
                                },
                            });
                        }
                    }
                }
                search_pos = abs_close + 1;
            } else {
                search_pos = content_start;
            }
        }
    }

    locations
}
