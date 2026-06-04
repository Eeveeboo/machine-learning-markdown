// ---------------------------------------------------------------------------
// Linear (fully-connected) block.
//
// Port of src/plugins/builtins/Linear.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct LinearBlockDef;

impl BlockDef for LinearBlockDef {
    fn name(&self) -> &str {
        "Linear"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![ParamSpec {
                name: "out_features".into(),
                param_type: ParamType::Number,
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
            return Err("Linear requires an input".to_string());
        }
        let out_features = get_num(params, "out_features").ok_or("Missing out_features")? as usize;
        let inp = &inputs[0];
        let mut result = inp.clone();
        let last = result.len().saturating_sub(1);
        result[last] = out_features;
        Ok(vec![result])
    }

    fn param_count(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        let in_f = if !inputs.is_empty() && !inputs[0].is_empty() {
            inputs[0][inputs[0].len() - 1]
        } else {
            0
        };
        let out_f = get_num(params, "out_features").ok_or(0).unwrap_or(0.0) as usize;
        Some(in_f * out_f + out_f)
    }

    fn show_depth(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(LinearBlockDef));

    register_block_codegen("Linear", "pytorch", pytorch_codegen);
    register_block_codegen("Linear", "keras", keras_codegen);
    register_block_codegen("Linear", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen helpers
// ---------------------------------------------------------------------------

fn get_out_f(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("out_features") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("units") {
        n.value as usize
    } else if let Some(shape) = block.output_shapes.first() {
        shape.last().copied().unwrap_or(0)
    } else {
        0
    }
}

fn get_in_f(block: &Block) -> usize {
    if let Some(shape) = block.input_shapes.first() {
        shape.last().copied().unwrap_or(0)
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let in_f = get_in_f(block);
    let out_f = get_out_f(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.Linear({}, {})",
            block.id, in_f, out_f
        ))),
        forward: format!(
            "{} = self.{}({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn keras_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let out_f = get_out_f(block);
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.Dense({})({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            out_f,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let in_f = get_in_f(block);
    let out_f = get_out_f(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::Linear", block.id),
            body: format!(
                "let {} = candle_nn::linear({}, {}, vb.pp(\"{}\"))?;",
                block.id, in_f, out_f, block.id
            ),
        })),
        forward: format!(
            "{} = self.{}.forward(&{})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            block.id,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_block_def() {
        let def = LinearBlockDef;
        assert_eq!(def.name(), "Linear");

        let mut p = HashMap::new();
        p.insert(
            "out_features".to_string(),
            ParamValue::Number(Box::new(mlmd_core::ast::nodes::NumberVal::new(
                10.0,
                mlmd_core::ast::nodes::SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def
            .infer_shape(&[vec![3, 64]], &p)
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 10]]);

        assert_eq!(def.param_count(&[vec![3, 64]], &p), Some(64 * 10 + 10));
    }
}
