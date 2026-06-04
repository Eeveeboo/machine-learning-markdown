// ---------------------------------------------------------------------------
// Embedding block.
//
// Port of src/plugins/builtins/Embedding.ts
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::types::*;

use crate::helpers::get_num;

// ---------------------------------------------------------------------------
// BlockDef for shape inference
// ---------------------------------------------------------------------------

struct EmbeddingBlockDef;

impl BlockDef for EmbeddingBlockDef {
    fn name(&self) -> &str {
        "Embedding"
    }

    fn params(&self) -> &[ParamSpec] {
        static PARAMS: std::sync::LazyLock<Vec<ParamSpec>> = std::sync::LazyLock::new(|| {
            vec![
                ParamSpec {
                    name: "vocab_size".into(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParamSpec {
                    name: "embed_dim".into(),
                    param_type: ParamType::Number,
                    required: true,
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
            return Err("Embedding requires an input".to_string());
        }
        let seq_len = inputs[0].first().copied().unwrap_or(0);
        let d = get_num(params, "embed_dim").ok_or("Missing embed_dim")? as usize;
        Ok(vec![vec![seq_len, d]])
    }

    fn param_count(
        &self,
        _inputs: &[Shape],
        params: &HashMap<String, ParamValue>,
    ) -> Option<usize> {
        let vocab = get_num(params, "vocab_size").ok_or(0.0).unwrap_or(0.0) as usize;
        let dim = get_num(params, "embed_dim").ok_or(0.0).unwrap_or(0.0) as usize;
        Some(vocab * dim)
    }

    fn show_depth(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register() {
    register_block(Box::new(EmbeddingBlockDef));

    register_block_codegen("Embedding", "pytorch", pytorch_codegen);
    register_block_codegen("Embedding", "keras", keras_codegen);
    register_block_codegen("Embedding", "candle", candle_codegen);
}

// ---------------------------------------------------------------------------
// Codegen helpers
// ---------------------------------------------------------------------------

fn get_vocab(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("vocab_size") {
        n.value as usize
    } else {
        0
    }
}

fn get_embed(block: &Block) -> usize {
    if let Some(ParamValue::Number(n)) = block.params.get("embed_dim") {
        n.value as usize
    } else if let Some(ParamValue::Number(n)) = block.params.get("embedding_dim") {
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
    let vocab = get_vocab(block);
    let embed = get_embed(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Plain(format!(
            "self.{} = nn.Embedding({}, {})",
            block.id, vocab, embed
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
    let vocab = get_vocab(block);
    let embed = get_embed(block);
    BlockCodegenResult {
        init: None,
        forward: format!(
            "{} = keras.layers.Embedding({}, {})({})",
            output_vars.first().map(|s| s.as_str()).unwrap_or("?"),
            vocab,
            embed,
            input_vars.first().map(|s| s.as_str()).unwrap_or("?"),
        ),
    }
}

fn candle_codegen(
    block: &Block,
    input_vars: &[String],
    output_vars: &[String],
) -> BlockCodegenResult {
    let vocab = get_vocab(block);
    let dim = get_embed(block);
    BlockCodegenResult {
        init: Some(CandleInitOrString::Candle(CandleInit {
            field: format!("{}: candle_nn::Embedding", block.id),
            body: format!(
                "let {} = candle_nn::embedding({}, {}, vb.pp(\"{}\"))?;",
                block.id, vocab, dim, block.id
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
    fn test_embedding_block_def() {
        let def = EmbeddingBlockDef;
        assert_eq!(def.name(), "Embedding");

        let mut p = HashMap::new();
        p.insert(
            "vocab_size".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                1000.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        p.insert(
            "embed_dim".to_string(),
            ParamValue::Number(Box::new(NumberVal::new(
                128.0,
                SourceLoc {
                    line: 0,
                    col: 0,
                    offset: 0,
                },
            ))),
        );
        let shapes = def.infer_shape(&[vec![10]], &p).unwrap();
        assert_eq!(shapes, vec![vec![10, 128]]);

        assert_eq!(def.param_count(&[], &p), Some(1000 * 128));
    }
}
