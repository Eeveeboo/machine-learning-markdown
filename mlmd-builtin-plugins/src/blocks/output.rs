// ---------------------------------------------------------------------------
// Output block – the terminal node of a computation graph.
//
// Port of src/plugins/builtins/Output.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Output;

impl Plugin for Output {
    fn params(&self) -> Vec<ParamSpec> {
        vec![]
    }
    fn name(&self) -> &'static str {
        "Output"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Output requires an input".to_string());
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
        _block: &Block,
        input_vars: &[String],
        _output_vars: &[String],
    ) -> Option<BlockCodegenResult> {
        match target {
            "pytorch" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "return {}",
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?")
                    ),
                }
            }),
            "keras" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "return {}",
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?")
                    ),
                }
            }),
            "candle" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "Ok({})",
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?")
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(Output);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_block_def() {
        let def = Output;
        assert_eq!(def.name(), "Output");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
