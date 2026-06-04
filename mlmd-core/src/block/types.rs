use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ast::graph::Shape;
use crate::ast::nodes::ParamValue;

// ---------------------------------------------------------------------------
// ParamType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamType {
    Number,
    String,
    Bool,
    Shape,
    List,
    Bareword,
}

impl ParamType {
    /// Parse a ParamType from a string slice.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "number" => Some(ParamType::Number),
            "string" => Some(ParamType::String),
            "bool" => Some(ParamType::Bool),
            "shape" => Some(ParamType::Shape),
            "list" => Some(ParamType::List),
            "bareword" => Some(ParamType::Bareword),
            _ => None,
        }
    }

    /// Return the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            ParamType::Number => "number",
            ParamType::String => "string",
            ParamType::Bool => "bool",
            ParamType::Shape => "shape",
            ParamType::List => "list",
            ParamType::Bareword => "bareword",
        }
    }
}

// ---------------------------------------------------------------------------
// ParamSpec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpec {
    pub name: String,
    pub param_type: ParamType,
    pub required: bool,
    pub default: Option<ParamValue>,
}

// ---------------------------------------------------------------------------
// BlockDef trait
// ---------------------------------------------------------------------------

/// Trait that every block type must implement.
pub trait BlockDef: Send + Sync {
    /// The block type name, e.g. "Conv2d", "Linear".
    fn name(&self) -> &str;

    /// The parameter specifications for this block type.
    fn params(&self) -> &[ParamSpec];

    /// Infer output shapes given input shapes and the resolved parameters.
    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String>;

    /// Optional: total parameter count for this block.
    fn param_count(
        &self,
        _inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        None
    }

    /// Whether block height should scale with channel depth in SVG layout.
    /// Set to `false` for passthrough blocks (activations, merges, etc.).
    fn show_depth(&self) -> bool {
        true
    }
}
