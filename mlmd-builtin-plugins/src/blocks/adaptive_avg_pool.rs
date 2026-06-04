// ---------------------------------------------------------------------------
// AdaptiveAvgPool block.
//
// Port of src/plugins/builtins/AdaptiveAvgPool.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

use crate::helpers::{get_num, get_num_list};

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct AdaptiveAvgPool;

impl Plugin for AdaptiveAvgPool {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::shape("size").required()]
    }
    fn name(&self) -> &'static str {
        "AdaptiveAvgPool"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("AdaptiveAvgPool requires an input".to_string());
        }
        if inputs[0].is_empty() {
            return Err("AdaptiveAvgPool input cannot be empty".to_string());
        }
        let c = inputs[0][0];
        let size = get_num_list(params, "size");
        if size.len() < 2 {
            return Err("AdaptiveAvgPool requires size=(H,W)".to_string());
        }
        Ok(vec![vec![c, size[0] as usize, size[1] as usize]])
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
                let (out_h, out_w) = if let Some(ParamValue::Shape(s)) = block.params.get("size") {
                    if s.dims.len() >= 2 {
                        (s.dims[0], s.dims[1])
                    } else {
                        (1, 1)
                    }
                } else if let Some(v) = get_num(&block.params, "output_size") {
                    let s = v as usize;
                    (s, s)
                } else {
                    (1, 1)
                };
                BlockCodegenResult {
                    init: Some(CandleInitOrString::Plain(format!(
                        "self.{} = nn.AdaptiveAvgPool2d(({}, {}))",
                        block.id, out_h, out_w
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
                let (out_h, out_w) = if let Some(ParamValue::Shape(s)) = block.params.get("size") {
                    if s.dims.len() >= 2 {
                        (s.dims[0], s.dims[1])
                    } else {
                        (1, 1)
                    }
                } else {
                    (1, 1)
                };
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.Lambda(lambda x: tf.image.resize(x, ({}, {})))({})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        out_h,
                        out_w,
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            "candle" => Some({
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = {}.mean_keepdim(candle_core::D::Minus1)?.mean_keepdim(candle_core::D::Minus2)?;",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(AdaptiveAvgPool);

#[cfg(test)]
mod tests {
    use super::*;
    use mlmd_core::ast::nodes::{ShapeVal, SourceLoc};

    #[test]
    fn test_adaptive_avg_pool_block_def() {
        let def = AdaptiveAvgPool;
        assert_eq!(def.name(), "AdaptiveAvgPool");

        let mut p = HashMap::new();
        p.insert(
            "size".to_string(),
            ParamValue::Shape(Box::new(ShapeVal::new(
                vec![7, 7],
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[vec![3, 224, 224]], &p).unwrap();
        assert_eq!(shapes, vec![vec![3, 7, 7]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
