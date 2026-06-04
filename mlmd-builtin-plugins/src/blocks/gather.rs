// ---------------------------------------------------------------------------
// Gather block.
//
// Port of src/plugins/builtins/Gather.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct GatherBlockDef;

impl BlockDef for GatherBlockDef {
    fn name(&self) -> &str {
        "Gather"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "axis".into(),
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
            return Err("Gather requires an input".to_string());
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
    register_block(Box::new(GatherBlockDef));

    register_block_codegen("Gather", "pytorch", pytorch_codegen);
    register_block_codegen("Gather", "keras", keras_codegen);
    register_block_codegen("Gather", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let axis = get_num(&block.params, "axis").unwrap_or(0.0) as isize;
    let idx = input_vars.get(1).cloned().unwrap_or_else(|| "index".to_string());
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = torch.gather({}, {}, {})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            axis,
            idx,
        ),
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let axis = get_num(&block.params, "axis").unwrap_or(0.0) as isize;
    let idx = input_vars.get(1).cloned().unwrap_or_else(|| "indices".to_string());
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = tf.gather({}, {}, axis={})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            idx,
            axis,
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let axis = get_num(&block.params, "axis").unwrap_or(0.0) as isize;
    let idx = input_vars.get(1).cloned().unwrap_or_else(|| "index".to_string());
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {}.gather(&{}, {})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            idx,
            axis,
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
    fn test_gather_block_def() {
        let def = GatherBlockDef;
        assert_eq!(def.name(), "Gather");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
