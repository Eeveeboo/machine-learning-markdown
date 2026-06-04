// ---------------------------------------------------------------------------
// RNN block.
//
// Port of src/plugins/builtins/RNN.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct RNNBlockDef;

impl BlockDef for RNNBlockDef {
    fn name(&self) -> &str {
        "RNN"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![
                ParamSpec {
                    name: "hidden_size".into(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParamSpec {
                    name: "num_layers".into(),
                    param_type: ParamType::Number,
                    required: false,
                    default: None,
                },
            ]
        });
        &PARAMS
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("RNN requires an input".to_string());
        }
        let seq = inputs[0].first().copied().unwrap_or(0);
        let h = get_num(params, "hidden_size").ok_or("Missing hidden_size")? as usize;
        Ok(vec![vec![seq, h]])
    }

    fn param_count(&self, inputs: &[Shape], params: &HashMap<String, ParamValue>) -> Option<usize> {
        let in_size = if !inputs.is_empty() && !inputs[0].is_empty() {
            inputs[0][inputs[0].len() - 1]
        } else {
            0
        };
        let h = get_num(params, "hidden_size").ok_or(0.0).unwrap_or(0.0) as usize;
        let l = get_num(params, "num_layers").unwrap_or(1.0) as usize;
        let first_layer = 1 * (in_size * h + h * h + 2 * h);
        let extra_layer = if l > 1 {
            (l - 1) * 1 * (h * h + h * h + 2 * h)
        } else {
            0
        };
        Some(first_layer + extra_layer)
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(RNNBlockDef));

    register_block_codegen("RNN", "pytorch", pytorch_codegen);
    register_block_codegen("RNN", "keras", keras_codegen);
    register_block_codegen("RNN", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen helpers
// ---------------------------------------------------------------------------

fn get_hidden(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("hidden_size") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("hidden") {
        n.value as usize
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
    let input_shape = block.input_shapes.first().cloned().unwrap_or_default();
    let input_size = input_shape.last().copied().unwrap_or(0);
    let hidden = get_hidden(block);
    let layers = get_num(&block.params, "num_layers").unwrap_or(1.0) as usize;
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.RNN({}, {}, num_layers={}, batch_first=True)",
            block.id, input_size, hidden, layers
        ))),
        forward: format!(
            "{} = self.{}({})[0]",
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
    let hidden = get_hidden(block);
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.SimpleRNN({}, return_sequences=True)({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            hidden,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let input_shape = block.input_shapes.first().cloned().unwrap_or_default();
    let input_size = input_shape.last().copied().unwrap_or(0);
    let hidden = get_hidden(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::Linear", block.id),
            body: format!(
                "let {} = /* RNN not natively supported in candle_nn */ candle_nn::linear({}, {}, vb.pp(\"{}\"))?;",
                block.id, input_size + hidden, hidden, block.id
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
    use mlmd_core::ast::nodes::{NumberVal, SourceLoc};

    #[test]
    fn test_rnn_block_def() {
        let def = RNNBlockDef;
        assert_eq!(def.name(), "RNN");

        let mut p = HashMap::new();
        p.insert(
            "hidden_size".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                128.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[vec![10, 64]], &p).unwrap();
        assert_eq!(shapes, vec![vec![10, 128]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
    }
}
