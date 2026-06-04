// ---------------------------------------------------------------------------
// Split block.
//
// Port of src/plugins/builtins/Split.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct SplitBlockDef;

impl BlockDef for SplitBlockDef {
    fn name(&self) -> &str {
        "Split"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![
                ParamSpec {
                    name: "chunks".into(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParamSpec {
                    name: "axis".into(),
                    param_type: ParamType::Number,
                    required: false,
                    default: None,
                },
            ]
        });
        &PARAMS
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Split requires an input".to_string());
        }
        let n = get_num(params, "chunks").ok_or("Missing chunks")? as usize;
        let axis = get_num(params, "axis").unwrap_or(0.0) as usize;
        let mut out_shape = inputs[0].clone();
        if axis < out_shape.len() {
            out_shape[axis] = out_shape[axis] / n;
        }
        Ok(vec![out_shape.clone(); n])
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
    register_block(Box::new(SplitBlockDef));

    register_block_codegen("Split", "pytorch", pytorch_codegen);
    register_block_codegen("Split", "keras", keras_codegen);
    register_block_codegen("Split", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let n = get_num(&block.params, "chunks").unwrap_or(2.0) as usize;
    let axis = get_num(&block.params, "axis").unwrap_or(0.0) as isize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = torch.chunk({}, {}, dim={})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            n,
            axis,
        ),
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let n = get_num(&block.params, "chunks").unwrap_or(2.0) as usize;
    let axis = get_num(&block.params, "axis").unwrap_or(0.0) as isize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = tf.split({}, {}, axis={})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            n,
            axis,
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let n = get_num(&block.params, "chunks").unwrap_or(2.0) as usize;
    let axis = get_num(&block.params, "axis").unwrap_or(0.0) as isize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {}.chunk({}, {})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            n,
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
    use mlmd_core::ast::nodes::{NumberVal, SourceLoc};

    #[test]
    fn test_split_block_def() {
        let def = SplitBlockDef;
        assert_eq!(def.name(), "Split");
        assert!(!def.show_depth());

        let mut p = HashMap::new();
        p.insert("chunks".to_string(), ParamValue::Number(Box::new(NumberVal::new(4.0, SourceLoc { line: 0, col: 0, offset: 0 }))));
        let shapes = def.infer_shape(&[vec![8, 224, 224]], &p).unwrap();
        assert_eq!(shapes.len(), 4);
        assert_eq!(shapes[0], vec![2, 224, 224]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
