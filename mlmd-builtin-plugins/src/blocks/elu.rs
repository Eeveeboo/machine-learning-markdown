// ---------------------------------------------------------------------------
// ELU activation block.
//
// Port of src/plugins/builtins/ELU.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct ELU;

impl Plugin for ELU {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::number("alpha").optional()]
    }
    fn name(&self) -> &'static str {
        "ELU"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("ELU requires an input".to_string());
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
                let alpha = get_num(&block.params, "alpha").unwrap_or(1.0);
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.ELU(alpha={})",
                        block.id, alpha
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
                let alpha = get_num(&block.params, "alpha").unwrap_or(1.0);
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.ELU(alpha={})({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        alpha,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                let alpha = get_num(&block.params, "alpha").unwrap_or(1.0);
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = {}.elu({})?;",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        alpha,
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(ELU);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elu_block_def() {
        let def = ELU;
        assert_eq!(def.name(), "ELU");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
