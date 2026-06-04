// ---------------------------------------------------------------------------
// Input block – the entry point of a computation graph.
//
// Port of src/plugins/builtins/Input.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::LazyLock;

use mlmd_core::types::*;

use crate::helpers::require_shape;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct InputBlockDef;

impl BlockDef for InputBlockDef {
    fn name(&self) -> &str {
        "Input"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: LazyLock<Vec<ParamSpec>> = LazyLock::new(|| {
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
        Ok(vec![require_shape(params, "shape")?])
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(InputBlockDef));

    register_block_codegen("Input", "pytorch", pytorch_codegen);
    register_block_codegen("Input", "keras", keras_codegen);
    register_block_codegen("Input", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    _block: &Block,
    input_vars: &[String],
    _output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: input_vars.first().cloned().unwrap_or_default(),
    }
}

fn keras_codegen(
    _block: &Block,
    _input_vars: &[String],
    _output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: String::new(),
    }
}

fn candle_codegen(
    _block: &Block,
    _input_vars: &[String],
    _output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_block_def() {
        let def = InputBlockDef;
        assert_eq!(def.name(), "Input");
        assert!(!def.show_depth());

        let params = def.params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "shape");
        assert_eq!(params[0].param_type, ParamType::Shape);
        assert!(params[0].required);

        // infer_shape: returns the shape param
        let mut p = HashMap::new();
        p.insert(
            "shape".to_string(),
            ParamValue::Shape(Box::new(mlmd_core::ast::nodes::ShapeVal::new(
                vec![3, 224, 224],
                mlmd_core::ast::nodes::SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[], &p).unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        // Missing shape param -> error
        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
    }
}
