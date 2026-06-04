// ---------------------------------------------------------------------------
// Concat (concatenation) block.
//
// Port of src/plugins/builtins/Concat.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_plugin_api::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct Concat;

impl Plugin for Concat {
    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::number("axis").optional()]
    }
    fn name(&self) -> &'static str {
        "Concat"
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Concat requires inputs".to_string());
        }
        let axis_raw = get_num(params, "axis").unwrap_or(1.0);
        let rank = inputs[0].len();
        if rank == 0 {
            return Err("Concat input cannot be scalar".to_string());
        }

        // Resolve negative axis to positive.
        let axis = if axis_raw < 0.0 {
            let a = axis_raw as isize;
            let r = rank as isize;
            let pos = r + a;
            if pos < 0 || pos >= r {
                return Err(format!(
                    "Concat axis {} out of bounds for rank {}",
                    axis_raw, rank
                ));
            }
            pos as usize
        } else {
            let a = axis_raw as usize;
            if a >= rank {
                return Err(format!(
                    "Concat axis {} out of bounds for rank {}",
                    a, rank
                ));
            }
            a
        };

        let mut out = inputs[0].clone();
        for i in 1..inputs.len() {
            if inputs[i].len() != rank {
                return Err(format!(
                    "Concat input {} has rank {} but expected rank {}",
                    i,
                    inputs[i].len(),
                    rank
                ));
            }
            for d in 0..rank {
                if d != axis && inputs[i][d] != out[d] {
                    return Err(format!(
                        "Concat dimension mismatch at dim {}: expected {} but input {} has {}",
                        d, out[d], i, inputs[i][d]
                    ));
                }
            }
            out[axis] += inputs[i][axis];
        }

        Ok(vec![out])
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

    fn num_inputs(&self) -> Option<usize> {
        None
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
                let dim = get_num(&block.params, "axis").unwrap_or(1.0) as isize;
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = torch.cat([{}], dim={})",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        input_vars.join(", "),
                        dim,
                    ),
                }
            }),
            "keras" => Some({
                let axis = get_num(&block.params, "axis").unwrap_or(-1.0) as isize;
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = keras.layers.Concatenate(axis={})([{}])",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        axis,
                        input_vars.join(", "),
                    ),
                }
            }),
            "candle" => Some({
                let tensors: Vec<String> = input_vars.iter().map(|v| format!("&{}", v)).collect();
                BlockCodegenResult {
                    init: None,
                    forward: format!(
                        "{} = Tensor::cat(&[{}], 1)?;",
                        output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                        tensors.join(", "),
                    ),
                }
            }),
            _ => None,
        }
    }
}

register_plugin!(Concat);

#[cfg(test)]
mod tests {
    use super::*;
    use mlmd_core::ast::nodes::{NumberVal, SourceLoc};

    #[test]
    fn test_concat_block_def() {
        let def = Concat;
        assert_eq!(def.name(), "Concat");
        assert!(!def.show_depth());

        // Basic concat along default axis=1
        let shapes = def
            .infer_shape(&[vec![3, 224, 224], vec![3, 112, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 336, 224]]);

        // Concat along axis=0
        let mut params_axis0 = HashMap::new();
        params_axis0.insert(
            "axis".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(0.0, loc()))),
        );
        let shapes = def
            .infer_shape(&[vec![3, 224, 224], vec![5, 224, 224]], &params_axis0)
            .unwrap();
        assert_eq!(shapes, vec![vec![8, 224, 224]]);

        // Concat along axis=-1 (same as axis=2)
        let mut params_neg = HashMap::new();
        params_neg.insert(
            "axis".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(-1.0, loc()))),
        );
        let shapes = def
            .infer_shape(
                &[vec![3, 224, 64], vec![3, 224, 128]],
                &params_neg,
            )
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 192]]);

        // Error: no inputs
        assert!(def.infer_shape(&[], &HashMap::new()).is_err());

        // Error: scalar input
        assert!(def.infer_shape(&[vec![]], &HashMap::new()).is_err());

        // Error: rank mismatch
        let err = def
            .infer_shape(&[vec![3, 224], vec![3, 112, 224]], &HashMap::new())
            .unwrap_err();
        assert!(err.contains("rank"));

        // Error: dimension mismatch (non-axis dims differ)
        let err = def
            .infer_shape(&[vec![3, 224, 224], vec![5, 112, 224]], &HashMap::new())
            .unwrap_err();
        assert!(err.contains("dimension mismatch"));

        // Error: axis out of bounds
        let mut params_bad_axis = HashMap::new();
        params_bad_axis.insert(
            "axis".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(5.0, loc()))),
        );
        let err = def
            .infer_shape(&[vec![3, 224, 224], vec![3, 112, 224]], &params_bad_axis)
            .unwrap_err();
        assert!(err.contains("out of bounds"));

        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }

    fn loc() -> SourceLoc {
        SourceLoc {
            line: 0,
            col: 0,
            offset: 0,
        }
    }
}
