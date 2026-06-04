// ---------------------------------------------------------------------------
// MaxPool block.
//
// Port of src/plugins/builtins/MaxPool.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// Helper: pool output size
// ---------------------------------------------------------------------------

fn pool_out(size: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    let numerator = (size as isize) + 2 * (padding as isize) - (kernel as isize);
    if numerator < 0 {
        0
    } else {
        (numerator as usize) / stride + 1
    }
}

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct MaxPoolBlockDef;

impl BlockDef for MaxPoolBlockDef {
    fn name(&self) -> &str {
        "MaxPool"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![
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
            return Err("MaxPool requires an input".to_string());
        }
        if inputs[0].len() < 3 {
            return Err("MaxPool input must have at least 3 dims (C, H, W)".to_string());
        }
        let c = inputs[0][0];
        let h = inputs[0][1];
        let w = inputs[0][2];
        let k = get_num(params, "kernel").ok_or("Missing kernel")? as usize;
        let s = get_num(params, "stride").unwrap_or(k as f64) as usize;
        let p = get_num(params, "padding").unwrap_or(0.0) as usize;
        Ok(vec![vec![c, pool_out(h, k, s, p), pool_out(w, k, s, p)]])
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
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(MaxPoolBlockDef));

    register_block_codegen("MaxPool", "pytorch", pytorch_codegen);
    register_block_codegen("MaxPool", "keras", keras_codegen);
    register_block_codegen("MaxPool", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen helpers
// ---------------------------------------------------------------------------

fn get_kernel_ps(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("kernel") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("kernel_size") {
        n.value as usize
    } else {
        2
    }
}

fn get_stride_ps(block: &Block, kernel: usize) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("stride") {
        n.value as usize
    } else {
        kernel
    }
}

fn get_padding_ps(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("padding") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("pad") {
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
    let kernel = get_kernel_ps(block);
    let stride = get_stride_ps(block, kernel);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.MaxPool2d({}, stride={})",
            block.id, kernel, stride
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
    let kernel = get_kernel_ps(block);
    let stride = get_stride_ps(block, kernel);
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.MaxPooling2D(pool_size={}, strides={})({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            kernel,
            stride,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let kernel = get_kernel_ps(block);
    let stride = get_stride_ps(block, kernel);
    let padding = get_padding_ps(block);
    let pad_prefix = if padding > 0 {
        format!(
            "{}.pad_with_zeros(2, {}, {})?.pad_with_zeros(3, {}, {})?",
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            padding, padding, padding, padding
        )
    } else {
        input_vars.first().cloned().unwrap_or_default()
    };
    let method = if stride != kernel {
        "max_pool2d_with_stride"
    } else {
        "max_pool2d"
    };
    let args = if stride != kernel {
        format!("{}, {}", kernel, stride)
    } else {
        format!("{}", kernel)
    };
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {}.{}({})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            pad_prefix,
            method,
            args,
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
    fn test_max_pool_block_def() {
        let def = MaxPoolBlockDef;
        assert_eq!(def.name(), "MaxPool");

        let mut p = HashMap::new();
        p.insert("kernel".to_string(), ParamValue::Number(Box::new(NumberVal::new(2.0, SourceLoc { line: 0, col: 0, offset: 0 }))));
        let shapes = def.infer_shape(&[vec![3, 224, 224]], &p).unwrap();
        assert_eq!(shapes, vec![vec![3, 112, 112]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
    }

    #[test]
    fn test_pool_out() {
        assert_eq!(pool_out(224, 2, 2, 0), 112);
        assert_eq!(pool_out(225, 3, 2, 1), 113);
    }
}
