// ---------------------------------------------------------------------------
// MatMul (matrix multiplication) block.
//
// Port of src/plugins/builtins/MatMul.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct MatMulBlockDef;

impl BlockDef for MatMulBlockDef {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn params(&self) -> &[ParamSpec] {
        &[]
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.len() < 2 {
            return Err("MatMul requires two inputs".to_string());
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let last_a = a.last().copied().unwrap_or(0);
        let last_b = b.last().copied().unwrap_or(0);
        let second_last_b = if b.len() >= 2 { b[b.len() - 2] } else { 0 };
        let needs_transpose = last_a == last_b && last_a != second_last_b;
        let n = if needs_transpose {
            if b.len() >= 2 { b[b.len() - 2] } else { 0 }
        } else {
            last_b
        };
        if a.len() == 2 && b.len() == 2 {
            return Ok(vec![vec![a[0], n]]);
        }
        // Batched: [..., M, K] * [..., K, N] = [..., M, N]
        let mut result: Vec<usize> = a[..a.len().saturating_sub(2)].to_vec();
        if a.len() >= 2 {
            result.push(a[a.len() - 2]); // M
        }
        result.push(n);
        Ok(vec![result])
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
    register_block(Box::new(MatMulBlockDef));

    register_block_codegen("MatMul", "pytorch", pytorch_codegen);
    register_block_codegen("MatMul", "keras", keras_codegen);
    register_block_codegen("MatMul", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen functions
// ---------------------------------------------------------------------------

fn pytorch_codegen(
    _block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = torch.matmul({}, {})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.get(1).map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn keras_codegen(
    _block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.Dot(axes=-1)([{}])",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.join(", "),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let shape_a = block.input_shapes.first().cloned().unwrap_or_default();
    let shape_b = block.input_shapes.get(1).cloned().unwrap_or_default();
    let needs_transpose = shape_a.len() >= 2
        && shape_b.len() >= 2
        && shape_a[shape_a.len() - 1] == shape_b[shape_b.len() - 1]
        && shape_a[shape_a.len() - 1] != shape_b[shape_b.len() - 2];
    let rhs = if needs_transpose {
        format!("{}.t()?", input_vars.get(1).map(|s| s.as_str()).unwrap_or("?"))
    } else {
        input_vars.get(1).cloned().unwrap_or_default()
    };
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = {}.matmul(&{})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            rhs,
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
    fn test_mat_mul_block_def() {
        let def = MatMulBlockDef;
        assert_eq!(def.name(), "MatMul");
        assert!(!def.show_depth());

        // 2D x 2D
        let shapes = def
            .infer_shape(&[vec![3, 4], vec![4, 5]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 5]]);

        assert!(def.infer_shape(&[vec![3, 4]], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
