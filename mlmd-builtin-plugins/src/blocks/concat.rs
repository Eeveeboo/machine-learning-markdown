// ---------------------------------------------------------------------------
// Concat (concatenation) block.
//
// Port of src/plugins/builtins/Concat.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct ConcatBlockDef;

impl BlockDef for ConcatBlockDef {
    fn name(&self) -> &str {
        "Concat"
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
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Concat requires inputs".to_string());
        }
        let axis = get_num(params, "axis").unwrap_or(1.0) as usize;
        let mut out = inputs[0].clone();
        for i in 1..inputs.len() {
            if axis < out.len() && axis < inputs[i].len() {
                out[axis] += inputs[i][axis];
            }
        }
        Ok(vec![out])
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
    register_block(Box::new(ConcatBlockDef));

    register_block_codegen("Concat", "pytorch", pytorch_codegen);
    register_block_codegen("Concat", "keras", keras_codegen);
    register_block_codegen("Concat", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let dim = get_num(&block.params, "axis").unwrap_or(1.0) as isize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = torch.cat([{}], dim={})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.join(", "),
            dim,
        ),
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let axis = get_num(&block.params, "axis").unwrap_or(-1.0) as isize;
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.Concatenate(axis={})([{}])",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            axis,
            input_vars.join(", "),
        ),
    }
}

fn candle_codegen(
    _block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let tensors: Vec<String> = input_vars.iter().map(|v| format!("&{}", v)).collect();
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = Tensor::cat(&[{}], 1)?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            tensors.join(", "),
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
    fn test_concat_block_def() {
        let def = ConcatBlockDef;
        assert_eq!(def.name(), "Concat");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224], vec![3, 112, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 336, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
