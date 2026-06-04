// ---------------------------------------------------------------------------
// mlmd-example-plugin – reference plugin implementation.
//
// This crate demonstrates how to create a custom NNML plugin:
//   1. Define a struct that implements `BlockDef` (shape inference)
//   2. Implement codegen functions for pytorch / keras / candle
//   3. Provide a `register()` function that wires everything into the
//      global registries
//   4. Export a `MLMD_PLUGIN` C-ABI symbol for dynamic loading (ffi.rs)
//
// The block: **Scale** – element-wise multiplication by a learned factor.
//   - Params:  `factor` (Number, required) – the multiplier
//   - Shape:   passthrough (input shape = output shape)
//   - Codegen: `out = in * factor`
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

pub mod ffi;

// ---------------------------------------------------------------------------
// BlockDef – shape inference
// ---------------------------------------------------------------------------

struct ScaleBlockDef;

impl BlockDef for ScaleBlockDef {
    fn name(&self) -> &str {
        "Scale"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "factor".into(),
                param_type: ParamType::Number,
                required: true,
                default: None,
            }]
        });
        &PARAMS
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Scale requires at least one input".to_string());
        }
        // Scale does not change the shape — passthrough
        Ok(vec![inputs[0].clone()])
    }

    fn param_count(
        &self,
        _inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        // Scale has exactly 1 learnable parameter (the factor)
        Some(1)
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(ScaleBlockDef));

    register_block_codegen("Scale", "pytorch", pytorch_codegen);
    register_block_codegen("Scale", "keras", keras_codegen);
    register_block_codegen("Scale", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen – PyTorch
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let factor = block
        .params
        .get("factor")
        .and_then(|v| {
            if let ParamValue::Number(n) = v {
                Some(n.value)
            } else {
                None
            }
        })
        .unwrap_or(1.0);

    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.Parameter(torch.tensor({factor}))",
            block.id,
        ))),
        forward: format!(
            "{} = self.{}({}) * self.{}",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
        ),
    }
}

// ---------------------------------------------------------------------------
// Codegen – Keras
// ---------------------------------------------------------------------------

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let factor = block
        .params
        .get("factor")
        .and_then(|v| {
            if let ParamValue::Number(n) = v {
                Some(n.value)
            } else {
                None
            }
        })
        .unwrap_or(1.0);

    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {} * {}",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            factor,
        ),
    }
}

// ---------------------------------------------------------------------------
// Codegen – Candle (Rust)
// ---------------------------------------------------------------------------

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let factor = block
        .params
        .get("factor")
        .and_then(|v| {
            if let ParamValue::Number(n) = v {
                Some(n.value)
            } else {
                None
            }
        })
        .unwrap_or(1.0);

    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: f64", block.id),
            body: format!("{}: {factor},", block.id),
        })),
        forward: format!(
            "{} = {}.mul(self.{})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_block_def() {
        let def = ScaleBlockDef;
        assert_eq!(def.name(), "Scale");
        assert!(!def.show_depth());

        let params = def.params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "factor");
        assert_eq!(params[0].param_type, ParamType::Number);
        assert!(params[0].required);

        // infer_shape: passthrough
        let shapes = def
            .infer_shape(&[vec![1, 3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![1, 3, 224, 224]]);

        // No input -> error
        assert!(def.infer_shape(&[], &HashMap::new()).is_err());

        // param_count
        assert_eq!(def.param_count(&[vec![1, 3, 224, 224]], &HashMap::new()), Some(1));
    }

    #[test]
    fn test_scale_register_and_lookup() {
        register();
        let lookup = lookup_block("Scale");
        assert!(lookup.is_some(), "Scale should be registered");
        assert_eq!(lookup.unwrap().name(), "Scale");
    }
}
