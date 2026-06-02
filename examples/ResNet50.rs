use candle_core::{Result, Tensor};
use candle_nn::Module;

pub struct ResNet50 {
    Conv2d_0: candle_nn::Conv2d,
    BatchNorm_0: candle_nn::BatchNorm,
    Conv2d_1: candle_nn::Conv2d,
    BatchNorm_1: candle_nn::BatchNorm,
    Conv2d_2: candle_nn::Conv2d,
    BatchNorm_2: candle_nn::BatchNorm,
    Conv2d_3: candle_nn::Conv2d,
    BatchNorm_3: candle_nn::BatchNorm,
}

impl ResNet50 {
    pub fn new(
        Conv2d_0: candle_nn::Conv2d,
        BatchNorm_0: candle_nn::BatchNorm,
        Conv2d_1: candle_nn::Conv2d,
        BatchNorm_1: candle_nn::BatchNorm,
        Conv2d_2: candle_nn::Conv2d,
        BatchNorm_2: candle_nn::BatchNorm,
        Conv2d_3: candle_nn::Conv2d,
        BatchNorm_3: candle_nn::BatchNorm,
    ) -> Self {
        Self {
            Conv2d_0,
            BatchNorm_0,
            Conv2d_1,
            BatchNorm_1,
            Conv2d_2,
            BatchNorm_2,
            Conv2d_3,
            BatchNorm_3,
        }
    }

    pub fn forward(&self,
        skip: &Tensor
    ) -> Result<Tensor> {
        let x = self.Conv2d_0.forward(&skip)?;
        let x = self.BatchNorm_0.forward(&x)?;
        let x = x.relu()?;
        let x = self.Conv2d_1.forward(&x)?;
        let x = self.BatchNorm_1.forward(&x)?;
        let x = x.relu()?;
        let x = self.Conv2d_2.forward(&x)?;
        let main_out = self.BatchNorm_2.forward(&x)?;
        let x = self.Conv2d_3.forward(&skip)?;
        let skip_out = self.BatchNorm_3.forward(&x)?;
        let x = (&main_out + &skip_out)?;
        let x = x.relu()?;
        Ok(x)
    }
}
