use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct UNet {
    Conv2d_0: candle_nn::Conv2d,
    Conv2d_1: candle_nn::Conv2d,
    Conv2d_2: candle_nn::Conv2d,
    Conv2d_3: candle_nn::Conv2d,
    Conv2d_4: candle_nn::Conv2d,
    Conv2d_5: candle_nn::Conv2d,
}

impl UNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let Conv2d_0 = candle_nn::conv2d(0, 32, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_0"))?;
        let Conv2d_1 = candle_nn::conv2d(0, 64, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_1"))?;
        let Conv2d_2 = candle_nn::conv2d(0, 128, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_2"))?;
        let Conv2d_3 = candle_nn::conv2d(0, 64, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_3"))?;
        let Conv2d_4 = candle_nn::conv2d(0, 32, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("Conv2d_4"))?;
        let Conv2d_5 = candle_nn::conv2d(0, 1, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("Conv2d_5"))?;
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
        let x = self.Conv2d_0.forward(&x)?;
        let enc1_skip = x.relu()?;
        let x = candle_nn::ops::max_pool2d(&enc1_skip, 2, 2)?;
        let x = self.Conv2d_1.forward(&x)?;
        let enc2_skip = x.relu()?;
        let x = candle_nn::ops::max_pool2d(&enc2_skip, 2, 2)?;
        let x = self.Conv2d_2.forward(&x)?;
        let bottleneck = x.relu()?;
        let up1 = unimplemented!("TransposedConv2d not supported in candle")  // bottleneck;
        let x = Tensor::cat(&[&up1, &enc2_skip], 1)?;
        let x = self.Conv2d_3.forward(&x)?;
        let dec1 = x.relu()?;
        let up2 = unimplemented!("TransposedConv2d not supported in candle")  // dec1;
        let x = Tensor::cat(&[&up2, &enc1_skip], 1)?;
        let x = self.Conv2d_4.forward(&x)?;
        let x = x.relu()?;
        let x = self.Conv2d_5.forward(&x)?;
        Ok(x)
    }
}
