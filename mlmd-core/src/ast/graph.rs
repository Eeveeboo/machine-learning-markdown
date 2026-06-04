use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ast::nodes::{ParamValue, SourceLoc};

// ---------------------------------------------------------------------------
// Shape type alias
// ---------------------------------------------------------------------------

pub type Shape = Vec<usize>;

// ---------------------------------------------------------------------------
// Block — a single computational node in the graph
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Unique generated id, e.g. "block_0"
    pub id: String,
    /// Layer type, e.g. "Conv2d"
    pub block_type: String,
    /// Resolved parameters
    pub params: HashMap<String, ParamValue>,
    /// Populated by shape inference
    pub input_shapes: Vec<Shape>,
    /// Populated by shape inference; array supports multi-output blocks
    pub output_shapes: Vec<Shape>,
    /// Populated by shape inference; total parameter count for this block
    pub param_count: Option<usize>,
    /// Whether block height should scale with channel depth in SVG
    pub show_depth: Option<bool>,
    /// Source location from the original source text
    pub loc: SourceLoc,
}

// ---------------------------------------------------------------------------
// Edge — a directed data-flow edge between two blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Block.id of the source block
    pub from: String,
    /// Block.id of the destination block
    pub to: String,
    /// Named tensor if this edge was created via `-> [name]`
    pub tensor_name: Option<String>,
    /// Populated by shape inference
    pub shape: Option<Shape>,
}

// ---------------------------------------------------------------------------
// Group — a logical grouping / namespace for a set of blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    /// Hierarchical path, e.g. ["ResNet50", "Bottleneck"]
    pub path: Vec<String>,
    /// The ids of blocks belonging to this group
    pub block_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// Graph — top-level directed graph IR produced after parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub blocks: Vec<Block>,
    pub edges: Vec<Edge>,
    pub groups: Vec<Group>,
}
