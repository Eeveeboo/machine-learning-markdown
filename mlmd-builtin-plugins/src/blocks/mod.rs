// ---------------------------------------------------------------------------
// Block module declarations.
// Each sub-module exports a public `register()` function that registers its
// `BlockDef` (for shape inference) and codegen functions with the global
// registries in `mlmd-core`.
// ---------------------------------------------------------------------------

pub mod input;
pub mod relu;

pub mod output;
pub mod linear;
pub mod embedding;
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod transposed_conv2d;
pub mod leaky_relu;
pub mod prelu;
pub mod elu;
pub mod selu;
pub mod gelu;
pub mod silu;
pub mod sigmoid;
pub mod tanh;
pub mod softmax;
pub mod batch_norm;
pub mod layer_norm;
pub mod group_norm;
pub mod instance_norm;
pub mod max_pool;
pub mod avg_pool;
pub mod global_avg_pool;
pub mod adaptive_avg_pool;
pub mod lstm;
pub mod gru;
pub mod rnn;
pub mod dropout;
pub mod flatten;
pub mod reshape;
pub mod pad;
pub mod add;
pub mod mul;
pub mod sub;
pub mod div;
pub mod concat;
pub mod mat_mul;
pub mod split;
pub mod repeat;
pub mod map;
pub mod gather;
