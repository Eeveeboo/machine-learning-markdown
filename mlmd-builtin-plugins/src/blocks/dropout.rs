// ---------------------------------------------------------------------------
// Dropout block.
//
// Port of src/plugins/builtins/Dropout.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Dropout;

impl Plugin for Dropout {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::number("p").optional()]
    }
    fn name(&self) -> &'static str {
        "Dropout"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Dropout requires an input".to_string());
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
                let p = get_p(block);
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.Dropout(p={})",
                        block.id, p
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
                let p = get_p(block);
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.Dropout({})({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        p,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                let p = get_p(block);
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Candle(CandleInit {
                        field: format!("{}: candle_nn::Dropout", block.id),
                        body: format!("let {} = candle_nn::Dropout::new({});", block.id, p),
                    })),
                    forward: format!(
                        "{} = self.{}.forward(&{}, true)?;",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        block.id,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(Dropout);

fn get_p(block: &Block) -> f64 {
    if let Some(ParamValue::Number(n)) = block.params.get("p") {
        n.value
    } else if let Some(ParamValue::Number(n)) = block.params.get("rate") {
        n.value
    } else {
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_block_def() {
        let def = Dropout;
        assert_eq!(def.name(), "Dropout");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
