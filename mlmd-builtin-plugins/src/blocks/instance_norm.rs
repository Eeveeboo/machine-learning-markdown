// ---------------------------------------------------------------------------
// InstanceNorm block.
//
// Port of src/plugins/builtins/InstanceNorm.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct InstanceNormBlockDef;

impl BlockDef for InstanceNormBlockDef {
    fn name(&self) -> &str {
        "InstanceNorm"
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
            return Err("InstanceNorm requires an input".to_string());
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
        Some(inputs[0][0] * 2)
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(InstanceNormBlockDef));

    register_block_codegen("InstanceNorm", "pytorch", pytorch_codegen);
    register_block_codegen("InstanceNorm", "keras", keras_codegen);
    register_block_codegen("InstanceNorm", "candle", candle_codegen);
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
    let dims = input_shape.len();
    let init_expr = if dims <= 2 {
        let num = input_shape.last().copied().unwrap_or(0);
        format!("self.{} = nn.InstanceNorm1d({})", block.id, num)
    } else {
        let num = if dims >= 3 {
            input_shape[input_shape.len() - 3]
        } else if dims >= 2 {
            input_shape[1]
        } else {
            0
        };
        format!("self.{} = nn.InstanceNorm2d({})", block.id, num)
    };
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(init_expr)),
        forward: format!(
            "{} = self.{}({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input_shape = block.input_shapes.first().cloned().unwrap_or_default();
    let channels = if input_shape.len() >= 3 {
        input_shape[input_shape.len() - 3]
    } else if input_shape.len() >= 2 {
        input_shape[1]
    } else {
        input_shape.last().copied().unwrap_or(0)
    };
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.GroupNormalization(groups={})({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            channels,
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
    let features = if input_shape.len() >= 4 {
        input_shape[input_shape.len() - 3]
    } else if input_shape.len() >= 3 {
        input_shape[input_shape.len() - 1]
    } else {
        input_shape.last().copied().unwrap_or(0)
    };
    // Uses BatchNorm as approximation for InstanceNorm in Candle
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::BatchNorm", block.id),
            body: format!(
                "let {} = candle_nn::batch_norm({}, 1e-5, vb.pp(\"{}\"))?;",
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
    fn test_instance_norm_block_def() {
        let def = InstanceNormBlockDef;
        assert_eq!(def.name(), "InstanceNorm");

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[vec![3, 224, 224]], &HashMap::new()), Some(6));
    }
}
