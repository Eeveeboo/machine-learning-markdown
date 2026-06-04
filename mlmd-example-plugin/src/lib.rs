// ---------------------------------------------------------------------------
// mlmd-example-plugin – reference plugin implementation.
//
// This crate demonstrates how to create a custom NNML plugin using the
// `mlmd-plugin-api` crate:
//   1. Define a unit struct
//   2. Implement the `Plugin` trait
//   3. Invoke `register_plugin!` to wire into the global registries
//   4. Invoke `export_plugin!` to generate the `MLMD_PLUGIN` FFI symbol
//
// The block: **Scale** – element-wise multiplication by a learned factor.
//   - Params:  `factor` (Number, required) – the multiplier
//   - Shape:   passthrough (input shape = output shape)
//   - Codegen: `out = in * factor`
// ---------------------------------------------------------------------------

use mlmd_plugin_api::*;

// ---------------------------------------------------------------------------
// Block definition
// ---------------------------------------------------------------------------

struct Scale;

impl Plugin for Scale {
    fn name(&self) -> &'static str {
        "Scale"
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::number("factor").required()]
    }

    fn infer_shape(
        &self,
        inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Result<Vec<Shape>, String> {
        if inputs.is_empty() {
            return Err("Scale requires at least one input".to_string());
        }
        // Scale does not change the shape — passthrough
        Ok(vec![inputs[0].clone()])
    }

    fn param_count(
        &self,
        _inputs: &[Shape],
        _params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        // Scale has exactly 1 learnable parameter (the factor)
        Some(1)
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
        let factor = block
            .params
            .get("factor")
            .and_then(|v| {
                if let ParamValue::Number(n) = v {
                    Some(n.value)
                } else {
                    None
                }
            })
            .unwrap_or(1.0);

        match target {
            "pytorch" => Some(BlockCodegenResult::stateful(
                format!("self.{} = nn.Parameter(torch.tensor({factor}))", block.id),
                format!(
                    "{} = self.{}({}) * self.{}",
                    output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    block.id,
                    input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    block.id,
                ),
            )),
            "keras" => Some(BlockCodegenResult::stateless(format!(
                "{} = {} * {factor}",
                output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            ))),
            "candle" => Some(BlockCodegenResult::candle(
                format!("{}: f64", block.id),
                format!("{}: {factor},", block.id),
                format!(
                    "{} = {}.mul(self.{})?;",
                    output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
                    block.id,
                ),
            )),
            _ => None,
        }
    }
}

register_plugin!(Scale);

// Export the MLMD_PLUGIN C-ABI symbol for dynamic loading.
// This generates all the FFI boilerplate automatically — no ffi.rs needed.
export_plugin!(Scale);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_plugin() {
        let plugin = Scale;
        assert_eq!(plugin.name(), "Scale");
        assert!(!plugin.show_depth());

        let params = plugin.params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "factor");
        assert_eq!(params[0].param_type, ParamType::Number);
        assert!(params[0].required);

        // infer_shape: passthrough
        let shapes = plugin
            .infer_shape(&[vec![1, 3, 224, 224]], &HashMap::new())
            .unwrap();
        assert_eq!(shapes, vec![vec![1, 3, 224, 224]]);

        // No input -> error
        assert!(plugin.infer_shape(&[], &HashMap::new()).is_err());

        // param_count
        assert_eq!(
            plugin.param_count(&[vec![1, 3, 224, 224]], &HashMap::new()),
            Some(1)
        );
    }

    #[test]
    fn test_scale_register_and_lookup() {
        // The generated register() function is called by the FFI setup,
        // but we can also call it directly in tests.
        register();
        let lookup = mlmd_core::block::registry::lookup_block("Scale");
        assert!(lookup.is_some(), "Scale should be registered");
        assert_eq!(lookup.unwrap().name(), "Scale");
    }

    #[test]
    fn test_scale_codegen_all_targets() {
        let plugin = Scale;
        let block = Block {
            id: "scale1".to_string(),
            block_type: "Scale".to_string(),
            params: HashMap::new(),
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shapes: vec![vec![1, 3, 224, 224]],
            param_count: None,
            show_depth: None,
            loc: mlmd_core::ast::nodes::SourceLoc {
                line: 0,
                col: 0,
                offset: 0,
            },
        };
        let input_vars = vec!["x".to_string()];
        let output_vars = vec!["out".to_string()];

        for target in ["pytorch", "keras", "candle"] {
            let result = plugin.codegen(target, &block, &input_vars, &output_vars);
            assert!(result.is_some(), "codegen for {target} should return Some");
        }

        // Unsupported target
        assert!(plugin
            .codegen("jax", &block, &input_vars, &output_vars)
            .is_none());
    }
}
