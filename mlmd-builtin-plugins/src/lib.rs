// ---------------------------------------------------------------------------
// mlmd-builtin-plugins – builtin block plugin implementations.
//
// This crate provides the standard library of neural-network block types
// (convolution, pooling, activation, normalisation, etc.) that ship with
// the NNML toolchain.
// ---------------------------------------------------------------------------

pub mod blocks;
pub mod ffi;
pub mod helpers;

/// Register all builtin block definitions and codegen functions.
pub fn register_all() {
    blocks::input::register();
    blocks::relu::register();

    blocks::output::register();
    blocks::linear::register();
    blocks::embedding::register();
    blocks::conv1d::register();
    blocks::conv2d::register();
    blocks::conv3d::register();
    blocks::transposed_conv2d::register();
    blocks::leaky_relu::register();
    blocks::prelu::register();
    blocks::elu::register();
    blocks::selu::register();
    blocks::gelu::register();
    blocks::silu::register();
    blocks::sigmoid::register();
    blocks::tanh::register();
    blocks::softmax::register();
    blocks::batch_norm::register();
    blocks::layer_norm::register();
    blocks::group_norm::register();
    blocks::instance_norm::register();
    blocks::max_pool::register();
    blocks::avg_pool::register();
    blocks::global_avg_pool::register();
    blocks::adaptive_avg_pool::register();
    blocks::lstm::register();
    blocks::gru::register();
    blocks::rnn::register();
    blocks::dropout::register();
    blocks::flatten::register();
    blocks::reshape::register();
    blocks::pad::register();
    blocks::add::register();
    blocks::mul::register();
    blocks::sub::register();
    blocks::div::register();
    blocks::concat::register();
    blocks::mat_mul::register();
    blocks::split::register();
    blocks::repeat::register();
    blocks::map::register();
    blocks::gather::register();
}
