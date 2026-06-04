use crate::ast::graph::Graph;

/// Rendering context for block plugins during SVG generation.
pub struct RenderContext {
    /// Total number of parameters across all blocks in the graph.
    pub total_params: usize,
    /// CSS class name for styling.
    pub class_name: String,
}

impl RenderContext {
    pub fn new(total_params: usize, class_name: &str) -> Self {
        Self {
            total_params,
            class_name: class_name.to_string(),
        }
    }
}

/// Code generation context for block plugins.
pub struct CodegenContext {
    /// Target language (e.g. "pytorch", "candle", "keras").
    pub target: String,
    /// The full block graph being code-generated.
    pub block_graph: Graph,
}

impl CodegenContext {
    pub fn new(target: &str, block_graph: Graph) -> Self {
        Self {
            target: target.to_string(),
            block_graph,
        }
    }
}
