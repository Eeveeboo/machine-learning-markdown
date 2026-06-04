// ---------------------------------------------------------------------------
// PReLU (Parametric ReLU) activation block.
//
// Port of src/plugins/builtins/PReLU.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct PReLU;

impl Plugin for PReLU {
    fn params(&self) -> Vec<ParamSpec> {
        vec![]
    }
    fn name(&self) -> &'static str {
        "PReLU"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("PReLU requires an input".to_string());
        }
        Ok(vec![inputs[0].clone()])
    }

    fn param_count(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        if inputs.is_empty() || inputs[0].is_empty() {
            return Some(0);
        }
        Some(inputs[0][0]) // one slope per input channel
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
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.PReLU()",
                        block.id
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
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.PReLU()({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = {}.relu()?;  /* PReLU: learnable slopes not supported in candle codegen */",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(PReLU);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelu_block_def() {
        let def = PReLU;
        assert_eq!(def.name(), "PReLU");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(
            def.param_count(&[vec![3, 224, 224]], &HashMap::new()),
            Some(3)
        );
    }
}
