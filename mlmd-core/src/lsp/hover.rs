// ---------------------------------------------------------------------------
// LSP hover — show block definition info when hovering over a block name
// ---------------------------------------------------------------------------

use tower_lsp::lsp_types::{
    Hover, HoverContents, MarkupContent, MarkupKind, Position,
};

use crate::block::registry::lookup_block;
use crate::lsp::word_at_position;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Get hover information at the given cursor position.
///
/// If the word under the cursor is a registered block type name, returns
/// formatted markdown with its parameter list.
pub fn get_hover(doc_source: &str, position: &Position) -> Option<Hover> {
    let lines: Vec<&str> = doc_source.lines().collect();
    let line = lines.get(position.line as usize)?;
    let word = word_at_position(line, position.character)?;

    // Look up the word in the block registry
    let def = lookup_block(&word)?;
    let params = def.params();

    // Format parameter list
    let param_lines: Vec<String> = params
        .iter()
        .map(|p| {
            let optional = if p.required { "" } else { "?" };
            format!("  `{}`: {}{}", p.name, p.param_type.as_str(), optional)
        })
        .collect();

    let param_section = if param_lines.is_empty() {
        "  _(none)_".to_string()
    } else {
        param_lines.join("\n")
    };

    let contents = format!(
        "**{}**\n```\nparams:\n{}\n```",
        def.name(),
        param_section
    );

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: contents,
        }),
        range: None,
    })
}
