// ---------------------------------------------------------------------------
// Block module declarations.
// Each sub-module exports a public `register()` function that registers its
// `BlockDef` (for shape inference) and codegen functions with the global
// registries in `mlmd-core`.
// ---------------------------------------------------------------------------

pub mod input;
pub mod relu;

pub mod adaptive_avg_pool;
pub mod add;
pub mod avg_pool;
pub mod batch_norm;
pub mod concat;
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod div;
pub mod dropout;
pub mod elu;
pub mod embedding;
pub mod flatten;
pub mod gather;
pub mod gelu;
pub mod global_avg_pool;
pub mod group_norm;
pub mod gru;
pub mod instance_norm;
pub mod layer_norm;
pub mod leaky_relu;
pub mod linear;
pub mod lstm;
pub mod map;
pub mod mat_mul;
pub mod max_pool;
pub mod mul;
pub mod output;
pub mod pad;
pub mod prelu;
pub mod repeat;
pub mod reshape;
pub mod rnn;
pub mod selu;
pub mod sigmoid;
pub mod silu;
pub mod softmax;
pub mod split;
pub mod sub;
pub mod tanh;
pub mod transposed_conv2d;
