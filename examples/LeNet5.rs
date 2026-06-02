use candle_core::{Result, Tensor};
use candle_nn::Module;

pub struct LeNet5 {
    Conv2d_0: candle_nn::Conv2d,
    Conv2d_1: candle_nn::Conv2d,
    Linear_0: candle_nn::Linear,
    Linear_1: candle_nn::Linear,
    Linear_2: candle_nn::Linear,
}

impl LeNet5 {
    pub fn new(
        Conv2d_0: candle_nn::Conv2d,
        Conv2d_1: candle_nn::Conv2d,
        Linear_0: candle_nn::Linear,
        Linear_1: candle_nn::Linear,
        Linear_2: candle_nn::Linear,
    ) -> Self {
        Self {
            Conv2d_0,
            Conv2d_1,
            Linear_0,
            Linear_1,
            Linear_2,
        }
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
