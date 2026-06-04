// ---------------------------------------------------------------------------
// Input block – the entry point of a computation graph.
//
// Port of src/plugins/builtins/Input.ts
// ---------------------------------------------------------------------------

use mlmd_plugin_api::*;
use std::collections::HashMap;

use crate::helpers::require_shape;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Input;

impl Plugin for Input {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::shape("shape").required()]
    }
    fn name(&self) -> &'static str {
        "Input"
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

    fn num_inputs(&self) -> Option<usize> {
        Some(0)
    }

    fn codegen(
        &self,
        target: &str,
        _block: &Block,
        input_vars: &[String],
        _output_vars: &[String],
    ) -> Option<BlockCodegenResult> {
        match target {
            "pytorch" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: input_vars.first().cloned().unwrap_or_default(),
                }
            }),
            "keras" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: String::new(),
                }
            }),
            "candle" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: String::new(),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(Input);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_block_def() {
        let def = Input;
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
