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
    /// Create model with auto-generated weight scope names.
    /// Use [UNet::with_scopes] for custom scope names.
    pub fn new(weights: VarBuilder) -> Result<Self> {
        Self::with_scopes(weights,
        "Conv2d_0",
        "Conv2d_1",
        "Conv2d_2",
        "Conv2d_3",
        "Conv2d_4",
        "Conv2d_5",
        )
    }

    /// Create model with custom weight-loading scope names.
    pub fn with_scopes(
        weights: VarBuilder,
        conv2d_0: &str,
        conv2d_1: &str,
        conv2d_2: &str,
        conv2d_3: &str,
        conv2d_4: &str,
        conv2d_5: &str,
    ) -> Result<Self> {
        let Conv2d_0 = candle_nn::conv2d(1, 32, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_0))?;
        let Conv2d_1 = candle_nn::conv2d(32, 64, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_1))?;
        let Conv2d_2 = candle_nn::conv2d(64, 128, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_2))?;
        let Conv2d_3 = candle_nn::conv2d(128, 64, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_3))?;
        let Conv2d_4 = candle_nn::conv2d(64, 32, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_4))?;
        let Conv2d_5 = candle_nn::conv2d(32, 1, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_5))?;
        Ok(Self {
            Conv2d_0,
            Conv2d_1,
            Conv2d_2,
            Conv2d_3,
            Conv2d_4,
            Conv2d_5,
        })
    }

    pub fn forward(&self,
        x: &Tensor
    ) -> Result<Tensor> {
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
