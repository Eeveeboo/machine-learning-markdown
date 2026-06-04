// ---------------------------------------------------------------------------
// Conv3d block.
//
// Port of src/plugins/builtins/Conv3d.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

use crate::helpers::{conv_out, get_num};

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Conv3d;

impl Plugin for Conv3d {
    fn params(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::number("filters").required(),
            ParamSpec::number("kernel").required(),
            ParamSpec::number("stride").optional(),
            ParamSpec::number("padding").optional(),
        ]
    }
    fn name(&self) -> &'static str {
        "Conv3d"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Conv3d requires an input".to_string());
        }
        if inputs[0].len() < 4 {
            return Err("Conv3d input must have at least 4 dims (C, D, H, W)".to_string());
        }
        let _c = inputs[0][0];
        let d = inputs[0][1];
        let h = inputs[0][2];
        let w = inputs[0][3];
        let f = get_num(params, "filters").ok_or("Missing filters")? as usize;
        let k = get_num(params, "kernel").ok_or("Missing kernel")? as usize;
        let s = get_num(params, "stride").unwrap_or(1.0) as usize;
        let p = get_num(params, "padding").unwrap_or(0.0) as usize;
        Ok(vec![vec![
            f,
            conv_out(d, k, s, p),
            conv_out(h, k, s, p),
            conv_out(w, k, s, p),
        ]])
    }

    fn param_count(&self, inputs: &[Shape], params: &HashMap<String, ParamValue>) -> Option<usize> {
        let in_c = if !inputs.is_empty() && !inputs[0].is_empty() {
            inputs[0][0]
        } else {
            0
        };
        let f = get_num(params, "filters").ok_or(0.0).unwrap_or(0.0) as usize;
        let k = get_num(params, "kernel").ok_or(0.0).unwrap_or(0.0) as usize;
        Some(in_c * f * k * k * k + f)
    }

    fn show_depth(&self) -> bool {
        true
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
                let in_ch = get_in_ch(block);
                let filters = get_filters(block);
                let kernel = get_kernel(block);
                let stride = get_stride(block);
                let padding = get_padding(block);
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.Conv3d({}, {}, {}, stride={}, padding={})",
                        block.id, in_ch, filters, kernel, stride, padding
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
                let filters = get_filters(block);
                let kernel = get_kernel(block);
                let stride = get_stride(block);
                let padding_str = if get_padding(block) == 0 {
                    "valid"
                } else {
                    "same"
                };
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.Conv3D({}, {}, strides={}, padding='{}')({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        filters,
                        kernel,
                        stride,
                        padding_str,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = unimplemented!(\"Conv3d not supported in candle\");  // {}",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(Conv3d);

fn get_in_ch(block: &Block) -> usize {
    if let Some(shape) = block.input_shapes.first() {
        if shape.len() >= 4 {
            shape[shape.len() - 4]
        } else if shape.len() >= 2 {
            shape[1]
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlmd_core::ast::nodes::{NumberVal, SourceLoc};

    #[test]
    fn test_conv3d_block_def() {
        let def = Conv3d;
        assert_eq!(def.name(), "Conv3d");

        let mut p = HashMap::new();
        p.insert(
            "filters".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                16.0,
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
        let shapes = def.infer_shape(&[vec![3, 16, 112, 112]], &p).unwrap();
        assert_eq!(shapes, vec![vec![16, 16, 112, 112]]);

        assert_eq!(
            def.param_count(&[vec![3, 16, 112, 112]], &p),
            Some(3 * 16 * 3 * 3 * 3 + 16)
        );
    }
}
