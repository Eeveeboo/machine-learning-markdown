use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ast::graph::{Block, Shape};
use crate::ast::nodes::ParamValue;
use crate::block::types::ParamSpec;

// ---------------------------------------------------------------------------
// CandleInit — init declaration for Candle (Rust) targets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleInit {
    /// Struct field declaration for statically-typed targets.
    pub field: String,
    /// Init body statement for statically-typed targets.
    pub body: String,
}

// ---------------------------------------------------------------------------
// CandleInitOrString — enum for the init field of BlockCodegenResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CandleInitOrString {
    Plain(String),
    Candle(CandleInit),
}

// ---------------------------------------------------------------------------
// BlockCodegenResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockCodegenResult {
    /// Optional init-time declaration.
    /// - `None` for stateless blocks (activations, merges, arithmetic).
    /// - `Some(CandleInitOrString::Plain(...))` for most targets.
    /// - `Some(CandleInitOrString::Candle(...))` for Candle backends.
    pub init: Option<CandleInitOrString>,
    /// The forward-pass code.
    pub forward: String,
}

impl BlockCodegenResult {
    /// Create a result for a **stateless** block (e.g. activations, merges).
    ///
    /// No init-time declaration is generated.
    pub fn stateless(forward: String) -> Self {
        BlockCodegenResult {
            init: None,
            forward,
        }
    }

    /// Create a result for a **stateful** block with a plain init string.
    ///
    /// Used for pytorch and keras targets.
    pub fn stateful(init: String, forward: String) -> Self {
        BlockCodegenResult {
            init: Some(CandleInitOrString::Plain(init)),
            forward,
        }
    }

    /// Create a result for a **Candle (Rust) target** block.
    ///
    /// * `field` — struct field declaration (e.g. `"linear: candle_nn::Linear"`)
    /// * `body` — init body statement (e.g. `"let linear = candle_nn::linear(...)?;"`)
    /// * `forward` — forward-pass code
    pub fn candle(field: String, body: String, forward: String) -> Self {
        BlockCodegenResult {
            init: Some(CandleInitOrString::Candle(CandleInit { field, body })),
            forward,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockCodegenFn — function pointer type for per-block codegen
// ---------------------------------------------------------------------------

/// Function pointer for generating code for a single block.
///
/// # Arguments
/// * `block` — the block to generate code for
/// * `input_vars` — names of the input tensor variables
/// * `output_vars` — names of the output tensor variables
pub type BlockCodegenFn =
    fn(block: &Block, input_vars: &[String], output_vars: &[String]) -> BlockCodegenResult;

// ---------------------------------------------------------------------------
// BlockPlugin trait
// ---------------------------------------------------------------------------

/// Interface that every plugin must satisfy.
pub trait BlockPlugin: Send + Sync {
    /// The block type name, e.g. "Conv2d", "Linear".
    fn name(&self) -> &str;

    /// Input tensor names for this block.
    fn inputs(&self) -> &[&str];

    /// Output tensor names for this block.
    fn outputs(&self) -> &[&str];

    /// Parameter specifications for this block type.
    fn params(&self) -> &[ParamSpec];

    /// Infer output shapes given input shapes and resolved parameters.
    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String>;

    /// Optional: total parameter count for this block.
    fn param_count(&self, inputs: &[Shape], params: &HashMap<String, ParamValue>) -> Option<usize>;

    /// Generate code for this block for a specific target.
    /// Returns `None` if this plugin does not support the given target.
    fn codegen(
        &self,
        target: &str,
        block: &Block,
        input_vars: &[String],
        output_vars: &[String],
    ) -> Option<BlockCodegenResult>;

    /// Whether block height should scale with channel depth in SVG layout.
    fn show_depth(&self) -> bool {
        true
    }
}
