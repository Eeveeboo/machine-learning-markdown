// ---------------------------------------------------------------------------
// Integration test: verify all 43 builtin block types are registered and
// that shape inference returns expected results for representative blocks.
// ---------------------------------------------------------------------------
//
// NOTE: Each test function releases the BlockDefGuard (mutex lock) before
// performing any assertion that could panic, to avoid poisoning the global
// registry mutex for subsequent tests.
// ---------------------------------------------------------------------------

use std::collections::HashMap;

use mlmd_core::ast::nodes::{NumberVal, ParamValue, ShapeVal, SourceLoc};
use mlmd_core::block::registry::lookup_block;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const LOC: SourceLoc = SourceLoc {
    line: 0,
    col: 0,
    offset: 0,
};

/// Build a params HashMap from (key, f64) pairs.
fn num_params(pairs: Vec<(&str, f64)>) -> HashMap<String, ParamValue> {
    pairs
        .into_iter()
        .map(|(k, v)| {
            (
                k.to_string(),
                ParamValue::Number(Box::new(NumberVal::new(v, LOC))),
            )
        })
        .collect()
}

/// Build a shape ParamValue from a list of dimensions.
fn shape_param(dims: Vec<usize>) -> ParamValue {
    ParamValue::Shape(Box::new(ShapeVal::new(dims, LOC)))
}

// ---------------------------------------------------------------------------
// 1. All 43 blocks are registered
// ---------------------------------------------------------------------------

#[test]
fn test_all_43_blocks_registered() {
    mlmd_builtin_plugins::register_all();

    let expected_blocks = [
        "Input",
        "Output",
        "Linear",
        "Embedding",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "TransposedConv2d",
        "ReLU",
        "LeakyReLU",
        "PReLU",
        "ELU",
        "SELU",
        "GELU",
        "SiLU",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "BatchNorm",
        "LayerNorm",
        "GroupNorm",
        "InstanceNorm",
        "MaxPool",
        "AvgPool",
        "GlobalAvgPool",
        "AdaptiveAvgPool",
        "LSTM",
        "GRU",
        "RNN",
        "Dropout",
        "Flatten",
        "Reshape",
        "Pad",
        "Add",
        "Mul",
        "Sub",
        "Div",
        "Concat",
        "MatMul",
        "Split",
        "Repeat",
        "Map",
        "Gather",
    ];

    assert_eq!(
        expected_blocks.len(),
        43,
        "Expected exactly 43 builtin block names"
    );

    for name in &expected_blocks {
        let found = {
            let def = lookup_block(name);
            def.is_some()
        }; // guard dropped here
        assert!(
            found,
            "Block '{}' is NOT registered in the BlockDef registry",
            name
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Detailed shape-inference tests for representative block types
// ---------------------------------------------------------------------------

#[test]
fn test_conv2d_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![
        ("filters", 32.0),
        ("kernel", 3.0),
        ("stride", 1.0),
        ("padding", 0.0),
    ]);

    // Conv2d(filters=32, kernel=3, stride=1, padding=0) on [3, 32, 32]
    // → [32, (32-3+0)/1+1, (32-3+0)/1+1] = [32, 30, 30]
    let (result, pc) = {
        let def = lookup_block("Conv2d").unwrap();
        let r = def.infer_shape(&[vec![3, 32, 32]], &params);
        let p = def.param_count(&[vec![3, 32, 32]], &params);
        (r, p)
    };
    assert!(
        result.is_ok(),
        "Conv2d infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![32, 30, 30]);

    // param_count = in_c * filters * k * k + filters
    assert_eq!(pc, Some(3 * 32 * 3 * 3 + 32));
}

#[test]
fn test_linear_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("out_features", 10.0)]);

    let (result, pc) = {
        let def = lookup_block("Linear").unwrap();
        let r = def.infer_shape(&[vec![3, 64]], &params);
        let p = def.param_count(&[vec![3, 64]], &params);
        (r, p)
    };
    assert!(
        result.is_ok(),
        "Linear infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 10]);
    // param_count = in_features * out_features + out_features
    assert_eq!(pc, Some(64 * 10 + 10));

    // Error on missing param
    let err_result = {
        let def = lookup_block("Linear").unwrap();
        def.infer_shape(&[vec![3, 64]], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_relu_shape_inference() {
    mlmd_builtin_plugins::register_all();

    // Passthrough
    let result = {
        let def = lookup_block("ReLU").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &HashMap::new())
    };
    assert!(result.is_ok());
    assert_eq!(result.unwrap()[0], vec![3, 224, 224]);

    // Error on empty inputs
    let err_result = {
        let def = lookup_block("ReLU").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_lstm_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("hidden_size", 128.0)]);

    let result = {
        let def = lookup_block("LSTM").unwrap();
        def.infer_shape(&[vec![10, 64]], &params)
    };
    assert!(
        result.is_ok(),
        "LSTM infer_shape failed: {:?}",
        result.err()
    );
    let shapes = result.unwrap();
    assert_eq!(shapes.len(), 2, "LSTM should produce 2 output shapes");
    assert_eq!(shapes[0], vec![10, 128]);
    assert_eq!(shapes[1], vec![128]);
}

#[test]
fn test_input_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let mut params = HashMap::new();
    params.insert("shape".to_string(), shape_param(vec![3, 224, 224]));

    let result = {
        let def = lookup_block("Input").unwrap();
        def.infer_shape(&[], &params)
    };
    assert!(
        result.is_ok(),
        "Input infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 224, 224]);

    // Error on missing shape param
    let err_result = {
        let def = lookup_block("Input").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_embedding_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("vocab_size", 1000.0), ("embed_dim", 128.0)]);

    let (result, pc) = {
        let def = lookup_block("Embedding").unwrap();
        let r = def.infer_shape(&[vec![10]], &params);
        let p = def.param_count(&[], &params);
        (r, p)
    };
    assert!(
        result.is_ok(),
        "Embedding infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![10, 128]);
    assert_eq!(pc, Some(1000 * 128));
}

#[test]
fn test_split_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("chunks", 4.0)]);

    let result = {
        let def = lookup_block("Split").unwrap();
        def.infer_shape(&[vec![8, 224, 224]], &params)
    };
    assert!(
        result.is_ok(),
        "Split infer_shape failed: {:?}",
        result.err()
    );
    let shapes = result.unwrap();
    assert_eq!(shapes.len(), 4);
    assert_eq!(shapes[0], vec![2, 224, 224]);
}

#[test]
fn test_concat_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc) = {
        let def = lookup_block("Concat").unwrap();
        let r = def.infer_shape(&[vec![3, 224, 224], vec![3, 112, 224]], &HashMap::new());
        let p = def.param_count(&[], &HashMap::new());
        (r, p)
    };
    assert!(
        result.is_ok(),
        "Concat infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 336, 224]);
    assert_eq!(pc, Some(0));

    let err_result = {
        let def = lookup_block("Concat").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_matmul_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc) = {
        let def = lookup_block("MatMul").unwrap();
        let r = def.infer_shape(&[vec![3, 4], vec![4, 5]], &HashMap::new());
        let p = def.param_count(&[], &HashMap::new());
        (r, p)
    };
    assert!(
        result.is_ok(),
        "MatMul infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 5]);
    assert_eq!(pc, Some(0));

    let err_result = {
        let def = lookup_block("MatMul").unwrap();
        def.infer_shape(&[vec![3, 4]], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_max_pool_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("kernel", 2.0)]);

    let result = {
        let def = lookup_block("MaxPool").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &params)
    };
    assert!(
        result.is_ok(),
        "MaxPool infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 112, 112]);
}

#[test]
fn test_flatten_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc) = {
        let def = lookup_block("Flatten").unwrap();
        let r = def.infer_shape(&[vec![3, 224, 224]], &HashMap::new());
        let p = def.param_count(&[], &HashMap::new());
        (r, p)
    };
    assert!(
        result.is_ok(),
        "Flatten infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3 * 224 * 224]);
    assert_eq!(pc, Some(0));

    let err_result = {
        let def = lookup_block("Flatten").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_dropout_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let result = {
        let def = lookup_block("Dropout").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &HashMap::new())
    };
    assert!(
        result.is_ok(),
        "Dropout infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 224, 224]);

    let err_result = {
        let def = lookup_block("Dropout").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_batch_norm_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc) = {
        let def = lookup_block("BatchNorm").unwrap();
        let r = def.infer_shape(&[vec![3, 224, 224]], &HashMap::new());
        let p = def.param_count(&[vec![3, 224, 224]], &HashMap::new());
        (r, p)
    };
    assert!(
        result.is_ok(),
        "BatchNorm infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 224, 224]);
    // param_count = num_features * 2 (num_features from input[0][0] = 3)
    assert_eq!(pc, Some(6));

    let err_result = {
        let def = lookup_block("BatchNorm").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_layer_norm_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc) = {
        let def = lookup_block("LayerNorm").unwrap();
        let r = def.infer_shape(&[vec![3, 224, 224]], &HashMap::new());
        let p = def.param_count(&[vec![3, 224, 224]], &HashMap::new());
        (r, p)
    };
    assert!(
        result.is_ok(),
        "LayerNorm infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 224, 224]);
    // param_count = last_dim * 2 = 224 * 2
    assert_eq!(pc, Some(224 * 2));
}

#[test]
fn test_global_avg_pool_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let result = {
        let def = lookup_block("GlobalAvgPool").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &HashMap::new())
    };
    assert!(
        result.is_ok(),
        "GlobalAvgPool infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 1, 1]);

    let err_result = {
        let def = lookup_block("GlobalAvgPool").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_reshape_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let mut params = HashMap::new();
    params.insert("shape".to_string(), shape_param(vec![1, 28, 28]));

    let result = {
        let def = lookup_block("Reshape").unwrap();
        def.infer_shape(&[vec![784]], &params)
    };
    assert!(
        result.is_ok(),
        "Reshape infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![1, 28, 28]);

    let err_result = {
        let def = lookup_block("Reshape").unwrap();
        def.infer_shape(&[vec![784]], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_pad_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let mut params = HashMap::new();
    params.insert("padding".to_string(), shape_param(vec![1, 1, 1, 1]));

    let result = {
        let def = lookup_block("Pad").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &params)
    };
    assert!(result.is_ok(), "Pad infer_shape failed: {:?}", result.err());
    assert_eq!(result.unwrap()[0], vec![3, 226, 226]);

    let err_result = {
        let def = lookup_block("Pad").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_adaptive_avg_pool_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let mut params = HashMap::new();
    params.insert("size".to_string(), shape_param(vec![7, 7]));

    let result = {
        let def = lookup_block("AdaptiveAvgPool").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &params)
    };
    assert!(
        result.is_ok(),
        "AdaptiveAvgPool infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 7, 7]);

    let err_result = {
        let def = lookup_block("AdaptiveAvgPool").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_add_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc) = {
        let def = lookup_block("Add").unwrap();
        let r = def.infer_shape(&[vec![3, 224, 224], vec![3, 224, 224]], &HashMap::new());
        let p = def.param_count(&[], &HashMap::new());
        (r, p)
    };
    assert!(result.is_ok(), "Add infer_shape failed: {:?}", result.err());
    assert_eq!(result.unwrap()[0], vec![3, 224, 224]);
    assert_eq!(pc, Some(0));

    let err_result = {
        let def = lookup_block("Add").unwrap();
        def.infer_shape(&[], &HashMap::new())
    };
    assert!(err_result.is_err());
}

#[test]
fn test_mul_sub_div_sigmoid_tanh_softmax_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let passthrough_blocks = ["Mul", "Sub", "Div", "Sigmoid", "Tanh", "Softmax"];
    for name in &passthrough_blocks {
        let (result, pc, err_result) = {
            let def = lookup_block(name).unwrap();
            let r = def.infer_shape(&[vec![3, 224, 224]], &HashMap::new());
            let p = def.param_count(&[], &HashMap::new());
            let e = def.infer_shape(&[], &HashMap::new());
            (r, p, e)
        };
        assert!(
            result.is_ok(),
            "{} infer_shape failed: {:?}",
            name,
            result.err()
        );
        assert_eq!(
            result.unwrap()[0],
            vec![3, 224, 224],
            "{} should be passthrough",
            name
        );
        assert_eq!(pc, Some(0), "{} param_count should be 0", name);
        assert!(err_result.is_err(), "{} should error on empty inputs", name);
    }
}

#[test]
fn test_gru_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("hidden_size", 128.0)]);

    let result = {
        let def = lookup_block("GRU").unwrap();
        def.infer_shape(&[vec![10, 64]], &params)
    };
    assert!(result.is_ok(), "GRU infer_shape failed: {:?}", result.err());
    let shapes = result.unwrap();
    assert_eq!(shapes.len(), 2, "GRU should produce 2 output shapes");
    assert_eq!(shapes[0], vec![10, 128]);
    assert_eq!(shapes[1], vec![128]);
}

#[test]
fn test_rnn_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("hidden_size", 64.0)]);

    let result = {
        let def = lookup_block("RNN").unwrap();
        def.infer_shape(&[vec![10, 32]], &params)
    };
    assert!(result.is_ok(), "RNN infer_shape failed: {:?}", result.err());
    let shapes = result.unwrap();
    // RNN returns only 1 output shape (unlike LSTM/GRU which return 2)
    assert_eq!(shapes.len(), 1, "RNN should produce 1 output shape");
    assert_eq!(shapes[0], vec![10, 64]);
}

#[test]
fn test_softmax_param_count() {
    mlmd_builtin_plugins::register_all();

    let pc = {
        let def = lookup_block("Softmax").unwrap();
        def.param_count(&[], &HashMap::new())
    };
    assert_eq!(pc, Some(0));
}

#[test]
fn test_transposed_conv2d_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![
        ("filters", 16.0),
        ("kernel", 3.0),
        ("stride", 2.0),
        ("padding", 1.0),
    ]);

    // TransposedConv2d(filters=16, kernel=3, stride=2, padding=1) on [3, 4, 4]
    // conv_transpose_output_size(4, 3, 1, 2) = (4-1)*2 - 2*1 + 3 = 6 - 2 + 3 = 7
    let result = {
        let def = lookup_block("TransposedConv2d").unwrap();
        def.infer_shape(&[vec![3, 4, 4]], &params)
    };
    assert!(
        result.is_ok(),
        "TransposedConv2d infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![16, 7, 7]);
}

#[test]
fn test_conv1d_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("filters", 16.0), ("kernel", 3.0)]);

    // Conv1d(filters=16, kernel=3) on [3, 32] → [16, 30]  (padding=0, stride=1)
    let result = {
        let def = lookup_block("Conv1d").unwrap();
        def.infer_shape(&[vec![3, 32]], &params)
    };
    assert!(
        result.is_ok(),
        "Conv1d infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![16, 30]);
}

#[test]
fn test_repeat_map_gather_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let passthrough_blocks = ["Repeat", "Map", "Gather"];
    for name in &passthrough_blocks {
        let (result, err_result) = {
            let def = lookup_block(name).unwrap();
            let r = def.infer_shape(&[vec![3, 224, 224]], &HashMap::new());
            let e = def.infer_shape(&[], &HashMap::new());
            (r, e)
        };
        assert!(
            result.is_ok(),
            "{} infer_shape failed: {:?}",
            name,
            result.err()
        );
        assert_eq!(
            result.unwrap()[0],
            vec![3, 224, 224],
            "{} should be passthrough",
            name
        );
        assert!(err_result.is_err(), "{} should error on empty inputs", name);
    }
}

#[test]
fn test_all_activations_are_passthrough() {
    mlmd_builtin_plugins::register_all();

    let activations = [
        "LeakyReLU",
        "PReLU",
        "ELU",
        "SELU",
        "GELU",
        "SiLU",
        "Sigmoid",
        "Tanh",
    ];
    for name in &activations {
        let result = {
            let def = lookup_block(name).unwrap();
            def.infer_shape(&[vec![3, 224, 224]], &HashMap::new())
        };
        assert!(
            result.is_ok(),
            "{} infer_shape failed: {:?}",
            name,
            result.err()
        );
        assert_eq!(
            result.unwrap()[0],
            vec![3, 224, 224],
            "{} should be passthrough",
            name
        );
    }
}

#[test]
fn test_normalisation_blocks_are_passthrough() {
    mlmd_builtin_plugins::register_all();

    let norm_blocks = ["GroupNorm", "InstanceNorm"];
    for name in &norm_blocks {
        let result = {
            let def = lookup_block(name).unwrap();
            def.infer_shape(&[vec![3, 224, 224]], &HashMap::new())
        };
        assert!(
            result.is_ok(),
            "{} infer_shape failed: {:?}",
            name,
            result.err()
        );
        assert_eq!(
            result.unwrap()[0],
            vec![3, 224, 224],
            "{} should be passthrough",
            name
        );
    }
}

#[test]
fn test_pooling_blocks_shape_inference() {
    mlmd_builtin_plugins::register_all();

    // AvgPool(kernel=2) on [3, 224, 224] → [3, 112, 112]
    let params = num_params(vec![("kernel", 2.0)]);
    let result = {
        let def = lookup_block("AvgPool").unwrap();
        def.infer_shape(&[vec![3, 224, 224]], &params)
    };
    assert!(
        result.is_ok(),
        "AvgPool infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![3, 112, 112]);
}

#[test]
fn test_output_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let (result, pc, err_result) = {
        let def = lookup_block("Output").unwrap();
        let r = def.infer_shape(&[vec![10]], &HashMap::new());
        let p = def.param_count(&[], &HashMap::new());
        let e = def.infer_shape(&[], &HashMap::new());
        (r, p, e)
    };
    assert!(
        result.is_ok(),
        "Output infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![10]);
    assert_eq!(pc, Some(0));
    assert!(err_result.is_err());
}

#[test]
fn test_conv3d_shape_inference() {
    mlmd_builtin_plugins::register_all();

    let params = num_params(vec![("filters", 16.0), ("kernel", 3.0)]);

    // Conv3d(filters=16, kernel=3) on [3, 16, 32, 32] → [16, 14, 30, 30]
    // conv_out(16, 3, 1, 0) = (16-3+0)/1+1 = 14
    // conv_out(32, 3, 1, 0) = 30
    let result = {
        let def = lookup_block("Conv3d").unwrap();
        def.infer_shape(&[vec![3, 16, 32, 32]], &params)
    };
    assert!(
        result.is_ok(),
        "Conv3d infer_shape failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap()[0], vec![16, 14, 30, 30]);
}
