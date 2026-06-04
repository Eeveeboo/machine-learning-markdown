// ---------------------------------------------------------------------------
// mlmd-plugin-api — Clean plugin API for NNML.
//
// Inspired by `zed_extension_api`, this crate provides a single unified
// `Plugin` trait that plugin authors implement.  A single `register_plugin!`
// macro invocation generates all the boilerplate (BlockDef adapter, codegen
// dispatch functions, registration).
//
// # Usage
//
// ```ignore
// use mlmd_plugin_api::*;
// use std::collections::HashMap;
//
// struct Scale;
//
// impl Plugin for Scale {
//     fn name(&self) -> &'static str { "Scale" }
//
//     fn params(&self) -> Vec<ParamSpec> {
//         vec![ParamSpec::number("factor").required()]
//     }
//
//     fn infer_shape(&self, inputs: &[Shape], _params: &HashMap<String, ParamValue>) -> Result<Vec<Shape>, String> {
//         inputs.first().ok_or("Scale requires an input".into()).map(|s| vec![s.clone()])
//     }
//
//     fn codegen(&self, target: &str, block: &Block, input_vars: &[String], output_vars: &[String]) -> Option<BlockCodegenResult> {
//         match target {
//             "pytorch" => Some(BlockCodegenResult::stateful(
//                 format!("self.{} = nn.Parameter(...)", block.id),
//                 format!("{} = self.{}({})", output_vars[0], block.id, input_vars[0]),
//             )),
//             "keras" => Some(BlockCodegenResult::stateless(
//                 format!("{} = {} * factor", output_vars[0], input_vars[0]),
//             )),
//             "candle" => Some(BlockCodegenResult::candle(
//                 format!("{}: f64", block.id),
//                 format!("{}: factor,", block.id),
//                 format!("{} = {}.mul(self.{})?;", output_vars[0], input_vars[0], block.id),
//             )),
//             _ => None,
//         }
//     }
// }
//
// register_plugin!(Scale);
// ```
// ---------------------------------------------------------------------------

pub use mlmd_plugin_api_macros::{export_plugin, export_wasm_plugin, register_plugin};

// Re-export core types that plugin authors commonly need.
pub use mlmd_core::ast::graph::{Block, Shape};
pub use mlmd_core::ast::nodes::ParamValue;
pub use mlmd_core::block::types::{ParamSpec, ParamSpecBuilder, ParamType};
pub use mlmd_core::plugin::traits::{BlockCodegenResult, CandleInit, CandleInitOrString};

// Convenience: re-export HashMap since every Plugin impl needs it.
pub use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Plugin trait — the single interface every plugin implements
// ---------------------------------------------------------------------------

/// The unified plugin trait.
///
/// Implement this trait on a unit struct, then call
/// `register_plugin!(YourType)` to register it with the NNML system.
pub trait Plugin: Send + Sync {
    /// The block type name, e.g. `"Conv2d"`, `"ReLU"`.
    fn name(&self) -> &'static str;

    /// Parameter specifications for this block type.
    fn params(&self) -> Vec<ParamSpec>;

    /// Infer output shapes given input shapes and resolved parameter values.
    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String>;

    /// Optional: total number of learnable parameters for this block.
    fn param_count(
        &self,
        _inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        None
    }

    /// Whether the block height should scale with channel depth in SVG layout.
    /// Set to `false` for passthrough blocks (activations, merges, etc.).
    fn show_depth(&self) -> bool {
        true
    }

    /// Number of tensor inputs this block expects.
    ///
    /// Return `None` for a variable number (e.g. `Concat`).  Default: 1.
    fn num_inputs(&self) -> Option<usize> {
        Some(1)
    }

    /// Number of tensor outputs this block produces.
    ///
    /// Return `None` for a variable number (e.g. `Split`).  Default: 1.
    fn num_outputs(&self) -> Option<usize> {
        Some(1)
    }

    /// Generate code for the given target.
    ///
    /// Supported targets: `"pytorch"`, `"keras"`, `"candle"`.
    ///
    /// Return `Some(result)` for supported targets, `None` for unsupported ones.
    /// The generated dispatch function will **panic** if `None` is returned for a
    /// target that is registered (i.e., all 3 targets are expected to be supported
    /// for standard block types).
    fn codegen(
        &self,
        target: &str,
        block: &Block,
        input_vars: &[String],
        output_vars: &[String],
    ) -> Option<BlockCodegenResult>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Verify that re-exports work (compile-time check).
    #[test]
    fn test_param_spec_builder_reexported() {
        let ps = ParamSpec::number("out_features").required();
        assert_eq!(ps.name, "out_features");
        assert_eq!(ps.param_type, ParamType::Number);
        assert!(ps.required);
        assert!(ps.default.is_none());
    }
}
