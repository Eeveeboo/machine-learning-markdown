// ---------------------------------------------------------------------------
// LeakyReLU activation block.
//
// Port of src/plugins/builtins/LeakyReLU.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct LeakyReLU;

impl Plugin for LeakyReLU {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::number("negative_slope").optional()]
    }
    fn name(&self) -> &'static str {
        "LeakyReLU"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("LeakyReLU requires an input".to_string());
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
                let slope = get_num(&block.params, "negative_slope").unwrap_or(0.01);
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.LeakyReLU({})",
                        block.id, slope
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
                let slope = get_num(&block.params, "negative_slope").unwrap_or(0.01);
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.LeakyReLU(alpha={})({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        slope,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                let slope = get_num(&block.params, "negative_slope").unwrap_or(0.01);
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = {}.leaky_relu({})?;",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        slope,
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(LeakyReLU);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu_block_def() {
        let def = LeakyReLU;
        assert_eq!(def.name(), "LeakyReLU");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
