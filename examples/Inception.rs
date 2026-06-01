use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct Inception {
    Conv2d_0: candle_nn::Conv2d,
    Conv2d_1: candle_nn::Conv2d,
    Conv2d_2: candle_nn::Conv2d,
    Conv2d_3: candle_nn::Conv2d,
    Conv2d_4: candle_nn::Conv2d,
    Conv2d_5: candle_nn::Conv2d,
}

impl Inception {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let Conv2d_0 = candle_nn::conv2d(0, 64, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_0"))?;
        let Conv2d_1 = candle_nn::conv2d(0, 96, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_1"))?;
        let Conv2d_2 = candle_nn::conv2d(0, 128, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_2"))?;
        let Conv2d_3 = candle_nn::conv2d(0, 16, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_3"))?;
        let Conv2d_4 = candle_nn::conv2d(0, 32, 5, candle_nn::Conv2dConfig { stride: 1, padding: 2, ..Default::default() }, vb.pp("Conv2d_4"))?;
        let Conv2d_5 = candle_nn::conv2d(0, 32, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_5"))?;
        Ok(Self {
            Conv2d_0,
            Conv2d_1,
            Conv2d_2,
            Conv2d_3,
            Conv2d_4,
            Conv2d_5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.Conv2d_0.forward(&b4)?;
        let out1 = x.relu()?;
        let x = self.Conv2d_1.forward(&b4)?;
        let x = x.relu()?;
        let x = self.Conv2d_2.forward(&x)?;
        let out2 = x.relu()?;
        let x = self.Conv2d_3.forward(&b4)?;
        let x = x.relu()?;
        let x = self.Conv2d_4.forward(&x)?;
        let out3 = x.relu()?;
        let x = candle_nn::ops::max_pool2d(&b4, 3, 1)?;
        let x = self.Conv2d_5.forward(&x)?;
        let out4 = x.relu()?;
        let x = Tensor::cat(&[&out1, &out2, &out3, &out4], 1)?;
        Ok(x)
    }
}
