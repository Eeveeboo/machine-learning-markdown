pub mod candle;
pub mod keras;
pub mod pytorch;

/// Register all built-in codegen targets (pytorch, keras, candle).
/// Safe to call multiple times — each target uses `OnceLock` internally.
pub fn register_all_codegen_targets() {
    pytorch::PytorchCodegen::register();
    keras::KerasCodegen::register();
    candle::CandleCodegen::register();
}
