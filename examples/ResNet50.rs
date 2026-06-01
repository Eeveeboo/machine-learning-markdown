use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

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
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let Conv2d_0 = candle_nn::conv2d(0, 64, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_0"))?;
        let BatchNorm_0 = candle_nn::batch_norm(0, 1e-5, vb.pp("BatchNorm_0"))?;
        let Conv2d_1 = candle_nn::conv2d(0, 64, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_1"))?;
        let BatchNorm_1 = candle_nn::batch_norm(0, 1e-5, vb.pp("BatchNorm_1"))?;
        let Conv2d_2 = candle_nn::conv2d(0, 256, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_2"))?;
        let BatchNorm_2 = candle_nn::batch_norm(0, 1e-5, vb.pp("BatchNorm_2"))?;
        let Conv2d_3 = candle_nn::conv2d(0, 256, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_3"))?;
        let BatchNorm_3 = candle_nn::batch_norm(0, 1e-5, vb.pp("BatchNorm_3"))?;
        Ok(Self {
            Conv2d_0,
            BatchNorm_0,
            Conv2d_1,
            BatchNorm_1,
            Conv2d_2,
            BatchNorm_2,
            Conv2d_3,
            BatchNorm_3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
