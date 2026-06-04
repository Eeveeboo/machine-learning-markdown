// ---------------------------------------------------------------------------
// Conv1d block.
//
// Port of src/plugins/builtins/Conv1d.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::{conv_out, get_num};

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Conv1dBlockDef;

impl BlockDef for Conv1dBlockDef {
    fn name(&self) -> &str {
        "Conv1d"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![
                ParamSpec {
                    name: "filters".into(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParamSpec {
                    name: "kernel".into(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParamSpec {
                    name: "stride".into(),
                    param_type: ParamType::Number,
                    required: false,
                    default: None,
                },
                ParamSpec {
                    name: "padding".into(),
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
            return Err("Conv1d requires an input".to_string());
        }
        if inputs[0].len() < 2 {
            return Err("Conv1d input must have at least 2 dims (C, L)".to_string());
        }
        let _c = inputs[0][0];
        let l = inputs[0][1];
        let f = get_num(params, "filters").ok_or("Missing filters")? as usize;
        let k = get_num(params, "kernel").ok_or("Missing kernel")? as usize;
        let s = get_num(params, "stride").unwrap_or(1.0) as usize;
        let p = get_num(params, "padding").unwrap_or(0.0) as usize;
        Ok(vec![vec![f, conv_out(l, k, s, p)]])
    }

    fn param_count(&self, inputs: &[Shape], params: &HashMap<String, ParamValue>) -> Option<usize> {
        let in_c = if !inputs.is_empty() && !inputs[0].is_empty() {
            inputs[0][0]
        } else {
            0
        };
        let f = get_num(params, "filters").ok_or(0.0).unwrap_or(0.0) as usize;
        let k = get_num(params, "kernel").ok_or(0.0).unwrap_or(0.0) as usize;
        Some(in_c * f * k + f)
    }

    fn show_depth(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(Conv1dBlockDef));

    register_block_codegen("Conv1d", "pytorch", pytorch_codegen);
    register_block_codegen("Conv1d", "keras", keras_codegen);
    register_block_codegen("Conv1d", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen helpers
// ---------------------------------------------------------------------------

fn get_in_ch(block: &Block) -> usize {
    if let Some(shape) = block.input_shapes.first() {
        if shape.len() >= 2 {
            shape[shape.len() - 2]
        } else {
            shape.first().copied().unwrap_or(0)
        }
    } else {
        0
    }
}

fn get_filters(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("filters") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("out_channels") {
        n.value as usize
    } else {
        0
    }
}

fn get_kernel(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("kernel") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("kernel_size") {
        n.value as usize
    } else {
        3
    }
}

fn get_stride(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("stride") {
        n.value as usize
    } else {
        1
    }
}

fn get_padding(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("padding") {
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
    let in_ch = get_in_ch(block);
    let filters = get_filters(block);
    let kernel = get_kernel(block);
    let stride = get_stride(block);
    let padding = get_padding(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.Conv1d({}, {}, {}, stride={}, padding={})",
            block.id, in_ch, filters, kernel, stride, padding
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
    let filters = get_filters(block);
    let kernel = get_kernel(block);
    let stride = get_stride(block);
    let padding_num = get_padding(block);
    let padding_str = if padding_num == 0 { "valid" } else { "same" };
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.Conv1D({}, {}, strides={}, padding='{}')({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            filters,
            kernel,
            stride,
            padding_str,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let in_ch = get_in_ch(block);
    let filters = get_filters(block);
    let kernel = get_kernel(block);
    let stride = get_stride(block);
    let padding = get_padding(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::Conv1d", block.id),
            body: format!(
                "let {} = candle_nn::conv1d({}, {}, {}, candle_nn::Conv1dConfig {{ stride: {}, padding: {}, ..Default::default() }}, vb.pp(\"{}\"))?;",
                block.id, in_ch, filters, kernel, stride, padding, block.id
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
    fn test_conv1d_block_def() {
        let def = Conv1dBlockDef;
        assert_eq!(def.name(), "Conv1d");

        let mut p = HashMap::new();
        p.insert(
            "filters".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                32.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        p.insert(
            "kernel".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                3.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        p.insert(
            "stride".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                1.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        p.insert(
            "padding".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                1.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[vec![3, 224]], &p).unwrap();
        assert_eq!(shapes, vec![vec![32, 224]]);

        assert_eq!(def.param_count(&[vec![3, 224]], &p), Some(3 * 32 * 3 + 32));
    }
}
