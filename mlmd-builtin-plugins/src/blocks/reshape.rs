// ---------------------------------------------------------------------------
// Reshape block.
//
// Port of src/plugins/builtins/Reshape.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num_list;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct ReshapeBlockDef;

impl BlockDef for ReshapeBlockDef {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "shape".into(),
                param_type: ParamType::Shape,
                required: true,
                default: None,
            }]
        });
        &PARAMS
    }

    fn infer_shape(
        &self,
        _inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        let v = params.get("shape").ok_or("Reshape requires shape param")?;
        match v {
            ParamValue::Shape(s) => Ok(vec![s.dims.clone()]),
            ParamValue::List(l) => {
                let dims: Result<Vec<usize>, String> = l
                    .items
                    .iter()
                    .map(|i| {
                        if let ParamValue::Number(n) = i {
                            Ok(n.value as usize)
                        } else {
                            Err("Expected number in list for shape".to_string())
                        }
                    })
                    .collect();
                Ok(vec![dims?])
            }
            _ => Err("Reshape requires shape param".to_string()),
        }
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
    register_block(Box::new(ReshapeBlockDef));

    register_block_codegen("Reshape", "pytorch", pytorch_codegen);
    register_block_codegen("Reshape", "keras", keras_codegen);
    register_block_codegen("Reshape", "candle", candle_codegen);
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
    if let Some(ParamValue::Shape(s)) = block.params.get("shape") {
        let dims: Vec<String> = s.dims.iter().map(|d| d.to_string()).collect();
        BlockCodegenResult {
            init: None,
            forward: format!(
                "{} = {}.reshape({}.size(0), {})",
                output0,
                input0,
                input0,
                dims.join(", ")
            ),
        }
    } else {
        BlockCodegenResult {
            init: None,
            forward: format!("{} = {}.reshape({}.size(0), -1)", output0, input0, input0),
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
    if let Some(ParamValue::Shape(s)) = block.params.get("shape") {
        let dims: Vec<String> = s.dims.iter().map(|d| d.to_string()).collect();
        BlockCodegenResult {
            init: None,
            forward: format!(
                "{} = keras.layers.Reshape(({},))({})",
                output0,
                dims.join(", "),
                input0
            ),
        }
    } else {
        BlockCodegenResult {
            init: None,
            forward: format!("{} = keras.layers.Reshape((-1,))({})", output0, input0),
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
    let dims = get_num_list(&block.params, "shape");
    let shape_str = if dims.is_empty() {
        "0".to_string()
    } else {
        dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    BlockCodegenResult {
        init: None,
        forward: format!("{} = {}.reshape(&[{}])?;", output0, input0, shape_str),
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
    fn test_reshape_block_def() {
        let def = ReshapeBlockDef;
        assert_eq!(def.name(), "Reshape");
        assert!(!def.show_depth());

        let mut p = HashMap::new();
        p.insert(
            "shape".to_string(),
            ParamValue::Shape(Box::new(ShapeVal::new(
                vec![1, 28, 28],
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[vec![784]], &p).unwrap();
        assert_eq!(shapes, vec![vec![1, 28, 28]]);
    }
}
