// ---------------------------------------------------------------------------
// LayerNorm block.
//
// Port of src/plugins/builtins/LayerNorm.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct LayerNormBlockDef;

impl BlockDef for LayerNormBlockDef {
    fn name(&self) -> &str {
        "LayerNorm"
    }

    fn params(&self) -> &[ParamSpec] {
        &[]
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("LayerNorm requires an input".to_string());
        }
        Ok(vec![inputs[0].clone()])
    }

    fn param_count(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        if inputs.is_empty() || inputs[0].is_empty() {
            return Some(0);
        }
        Some(inputs[0][inputs[0].len() - 1] * 2)
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(LayerNormBlockDef));

    register_block_codegen("LayerNorm", "pytorch", pytorch_codegen);
    register_block_codegen("LayerNorm", "keras", keras_codegen);
    register_block_codegen("LayerNorm", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input_shape = block.input_shapes.first().cloned().unwrap_or_default();
    let normalized = if input_shape.len() > 1 {
        let rest: Vec<String> = input_shape[1..].iter().map(|d| d.to_string()).collect();
        format!("[{}]", rest.join(", "))
    } else {
        "[]".to_string()
    };
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.LayerNorm({})",
            block.id, normalized
        ))),
        forward: format!(
            "{} = self.{}({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn keras_codegen(
    _block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.LayerNormalization()({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input_shape = block.input_shapes.first().cloned().unwrap_or_default();
    let features = input_shape.last().copied().unwrap_or(0);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::LayerNorm", block.id),
            body: format!(
                "let {} = candle_nn::layer_norm({}, 1e-5, vb.pp(\"{}\"))?;",
                block.id, features, block.id
            ),
        })),
        forward: format!(
            "{} = self.{}.forward(&{})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
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
    fn test_layer_norm_block_def() {
        let def = LayerNormBlockDef;
        assert_eq!(def.name(), "LayerNorm");

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[vec![3, 224, 224]], &HashMap::new()), Some(224 * 2));
    }
}
