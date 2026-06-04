// ---------------------------------------------------------------------------
// Repeat block.
//
// Port of src/plugins/builtins/Repeat.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct RepeatBlockDef;

impl BlockDef for RepeatBlockDef {
    fn name(&self) -> &str {
        "Repeat"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "times".into(),
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
            return Err("Repeat requires an input".to_string());
        }
        Ok(vec![inputs[0].clone()])
    }

    fn param_count(
        &self,
        _inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        Some(0)
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(RepeatBlockDef));

    register_block_codegen("Repeat", "pytorch", pytorch_codegen);
    register_block_codegen("Repeat", "keras", keras_codegen);
    register_block_codegen("Repeat", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let times = get_num(&block.params, "times").unwrap_or(1.0) as usize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {}.repeat({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            times,
        ),
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let times = get_num(&block.params, "times").unwrap_or(1.0) as usize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.RepeatVector({})({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            times,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let times = get_num(&block.params, "times").unwrap_or(1.0) as usize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {}.repeat(&[{}])?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            times,
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
    fn test_repeat_block_def() {
        let def = RepeatBlockDef;
        assert_eq!(def.name(), "Repeat");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
