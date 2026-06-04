// ---------------------------------------------------------------------------
// Pad block.
//
// Port of src/plugins/builtins/Pad.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num_list;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct PadBlockDef;

impl BlockDef for PadBlockDef {
    fn name(&self) -> &str {
        "Pad"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "padding".into(),
                param_type: ParamType::Shape,
                required: true,
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
            return Err("Pad requires an input".to_string());
        }
        if inputs[0].len() < 3 {
            return Err("Pad input must have at least 3 dims (C, H, W)".to_string());
        }
        let c = inputs[0][0];
        let h = inputs[0][1];
        let w = inputs[0][2];
        let p = get_num_list(params, "padding");
        if p.len() < 4 {
            return Err("Pad requires padding=(P0,P1,P2,P3)".to_string());
        }
        Ok(vec![vec![
            c,
            h + p[0] as usize + p[1] as usize,
            w + p[2] as usize + p[3] as usize,
        ]])
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
    register_block(Box::new(PadBlockDef));

    register_block_codegen("Pad", "pytorch", pytorch_codegen);
    register_block_codegen("Pad", "keras", keras_codegen);
    register_block_codegen("Pad", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input0 = input_vars.first().map(|s| s.as_str()).unwrap_or("?");
    let output0 = output_vars.first().map(|s| s.as_str()).unwrap_or("?");
    if let Some(ParamValue::List(l)) = block.params.get("padding") {
        let vals: Vec<String> = l
            .items
            .iter()
            .filter_map(|i| {
                if let ParamValue::Number(n) = i {
                    Some(n.value.to_string())
                } else {
                    None
                }
            })
            .collect();
        BlockCodegenResult {
            init: None,
            forward: format!(
                "{} = torch.nn.functional.pad({}, ({}))",
                output0,
                input0,
                vals.join(", ")
            ),
        }
    } else {
        BlockCodegenResult {
            init: None,
            forward: format!("{} = torch.nn.functional.pad({}, (0, 0))", output0, input0),
        }
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input0 = input_vars.first().map(|s| s.as_str()).unwrap_or("?");
    let output0 = output_vars.first().map(|s| s.as_str()).unwrap_or("?");
    let p = get_num_list(&block.params, "padding");
    if p.len() >= 4 {
        BlockCodegenResult {
            init: None,
            forward: format!(
                "{} = keras.layers.ZeroPadding2D(padding=(({}, {}), ({}, {})))({})",
                output0, p[0], p[1], p[2], p[3], input0
            ),
        }
    } else {
        BlockCodegenResult {
            init: None,
            forward: format!("{} = keras.layers.ZeroPadding2D()({})", output0, input0),
        }
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input0 = input_vars.first().map(|s| s.as_str()).unwrap_or("?");
    let output0 = output_vars.first().map(|s| s.as_str()).unwrap_or("?");
    let p = get_num_list(&block.params, "padding");
    if p.len() >= 4 {
        BlockCodegenResult {
            init: None,
            forward: format!(
                "{} = {}.pad_with_zeros(2, {}, {})?.pad_with_zeros(3, {}, {})?",
                output0, input0, p[0], p[1], p[2], p[3]
            ),
        }
    } else {
        BlockCodegenResult {
            init: None,
            forward: format!("{} = {}.pad_with_zeros(2, 0, 0)?;", output0, input0),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlmd_core::ast::nodes::{ShapeVal, SourceLoc};

    #[test]
    fn test_pad_block_def() {
        let def = PadBlockDef;
        assert_eq!(def.name(), "Pad");
        assert!(!def.show_depth());

        let mut p = HashMap::new();
        p.insert(
            "padding".to_string(),
            ParamValue::Shape(Box::new(ShapeVal::new(
                vec![1, 1, 1, 1],
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[vec![3, 224, 224]], &p).unwrap();
        assert_eq!(shapes, vec![vec![3, 226, 226]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
    }
}
