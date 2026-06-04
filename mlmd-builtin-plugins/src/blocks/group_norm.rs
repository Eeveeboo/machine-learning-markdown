// ---------------------------------------------------------------------------
// GroupNorm block.
//
// Port of src/plugins/builtins/GroupNorm.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct GroupNormBlockDef;

impl BlockDef for GroupNormBlockDef {
    fn name(&self) -> &str {
        "GroupNorm"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "num_groups".into(),
                param_type: ParamType::Number,
                required: false,
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
            return Err("GroupNorm requires an input".to_string());
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
    register_block(Box::new(GroupNormBlockDef));

    register_block_codegen("GroupNorm", "pytorch", pytorch_codegen);
    register_block_codegen("GroupNorm", "keras", keras_codegen);
    register_block_codegen("GroupNorm", "candle", candle_codegen);
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
    let groups = get_num(&block.params, "num_groups").unwrap_or(32.0) as usize;
    let num = if input_shape.len() >= 3 {
        input_shape[input_shape.len() - 3]
    } else if input_shape.len() >= 2 {
        input_shape[1]
    } else {
        0
    };
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.GroupNorm({}, {})",
            block.id, groups, num
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
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let groups = get_num(&block.params, "num_groups").unwrap_or(32.0) as usize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.GroupNormalization(groups={})({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            groups,
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
    let groups = get_num(&block.params, "num_groups").unwrap_or(32.0) as usize;
    let channels = if input_shape.len() >= 3 {
        input_shape[input_shape.len() - 3]
    } else if input_shape.len() >= 2 {
        input_shape[1]
    } else {
        0
    };
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::GroupNorm", block.id),
            body: format!(
                "let {} = candle_nn::group_norm({}, {}, 1e-5, vb.pp(\"{}\"))?;",
                block.id, groups, channels, block.id
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
    fn test_group_norm_block_def() {
        let def = GroupNormBlockDef;
        assert_eq!(def.name(), "GroupNorm");

        let shapes = def
            .infer_shape(&[vec![6, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![6, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[vec![6, 224, 224]], &HashMap::new()), Some(12));
    }
}
