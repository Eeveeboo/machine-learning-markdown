// ---------------------------------------------------------------------------
// Softmax activation block.
//
// Port of src/plugins/builtins/Softmax.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Softmax;

impl Plugin for Softmax {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::number("dim").optional()]
    }
    fn name(&self) -> &'static str {
        "Softmax"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Softmax requires an input".to_string());
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

    fn codegen(
        &self,
        target: &str,
        block: &Block,
        input_vars: &[String],
        output_vars: &[String],
    ) -> Option<BlockCodegenResult> {
        match target {
            "pytorch" => Some({
                let dim = get_num(&block.params, "dim").unwrap_or(-1.0) as isize;
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.Softmax(dim={})",
                        block.id, dim
                    ))),
                    forward: format!(
                        "{} = self.{}({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        block.id,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "keras" => Some({
                let axis = get_num(&block.params, "dim").unwrap_or(-1.0) as isize;
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.Softmax(axis={})({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        axis,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                let dim = get_num(&block.params, "dim").unwrap_or(-1.0) as isize;
                let dim_expr = if dim == -1 {
                    "candle_core::D::Minus1".to_string()
                } else {
                    format!("{}", dim)
                };
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = candle_nn::ops::softmax(&{}, {})?;",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        dim_expr,
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(Softmax);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_block_def() {
        let def = Softmax;
        assert_eq!(def.name(), "Softmax");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
