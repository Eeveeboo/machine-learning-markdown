use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct LeNet5 {
    Conv2d_0: candle_nn::Conv2d,
    Conv2d_1: candle_nn::Conv2d,
    Linear_0: candle_nn::Linear,
    Linear_1: candle_nn::Linear,
    Linear_2: candle_nn::Linear,
}

impl LeNet5 {
    /// Create model with auto-generated weight scope names.
    /// Use [LeNet5::with_scopes] for custom scope names.
    pub fn new(weights: VarBuilder) -> Result<Self> {
        Self::with_scopes(weights,
        "Conv2d_0",
        "Conv2d_1",
        "Linear_0",
        "Linear_1",
        "Linear_2",
        )
    }

    /// Create model with custom weight-loading scope names.
    pub fn with_scopes(
        weights: VarBuilder,
        conv2d_0: &str,
        conv2d_1: &str,
        linear_0: &str,
        linear_1: &str,
        linear_2: &str,
    ) -> Result<Self> {
        let Conv2d_0 = candle_nn::conv2d(1, 6, 5, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_0))?;
        let Conv2d_1 = candle_nn::conv2d(6, 16, 5, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_1))?;
        let Linear_0 = candle_nn::linear(256, 120, weights.pp(linear_0))?;
        let Linear_1 = candle_nn::linear(120, 84, weights.pp(linear_1))?;
        let Linear_2 = candle_nn::linear(84, 10, weights.pp(linear_2))?;
        Ok(Self {
            Conv2d_0,
            Conv2d_1,
            Linear_0,
            Linear_1,
            Linear_2,
        })
    }

    pub fn forward(&self,
        x: &Tensor
    ) -> Result<Tensor> {
        let x = self.Conv2d_0.forward(&x)?;
        let x = x.relu()?;
        let x = candle_nn::ops::max_pool2d(&x, 2, 2)?;
        let x = self.Conv2d_1.forward(&x)?;
        let x = x.relu()?;
        let x = candle_nn::ops::max_pool2d(&x, 2, 2)?;
        let x = x.flatten_from(1)?;
        let x = self.Linear_0.forward(&x)?;
        let x = x.relu()?;
        let x = self.Linear_1.forward(&x)?;
        let x = x.relu()?;
        let x = self.Linear_2.forward(&x)?;
        Ok(x)
    }
}
