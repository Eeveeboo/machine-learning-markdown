// ---------------------------------------------------------------------------
// Mul (element-wise multiplication) block.
//
// Port of src/plugins/builtins/Mul.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct MulBlockDef;

impl BlockDef for MulBlockDef {
    fn name(&self) -> &str {
        "Mul"
    }

    fn params(&self) -> &[ParamSpec] {
        &[]
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Mul requires inputs".to_string());
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
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(MulBlockDef));

    register_block_codegen("Mul", "pytorch", pytorch_codegen);
    register_block_codegen("Mul", "keras", keras_codegen);
    register_block_codegen("Mul", "candle", candle_codegen);
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
            "{} = {}",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.join(" * "),
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
            "{} = keras.layers.Multiply()([{}])",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.join(", "),
        ),
    }
}

fn candle_codegen(
    _block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = (&{} * &{})?;",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            input_vars.get(1).map(|s| s.as_str()).unwrap_or("?"),
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
    fn test_mul_block_def() {
        let def = MulBlockDef;
        assert_eq!(def.name(), "Mul");
        assert!(!def.show_depth());

        let shapes = def
            .infer_shape(&[vec![3, 224, 224], vec![3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![3, 224, 224]]);

        assert!(def.infer_shape(&[], &HashMap::new()).is_err());
        assert_eq!(def.param_count(&[], &HashMap::new()), Some(0));
    }
}
