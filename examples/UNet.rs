use candle_core::{ModuleT, Result, Tensor};
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
        let x = enc1_skip.max_pool2d(2)?;
        let x = self.Conv2d_1.forward(&x)?;
        let enc2_skip = x.relu()?;
        let x = enc2_skip.max_pool2d(2)?;
        let x = self.Conv2d_2.forward(&x)?;
        let bottleneck = x.relu()?;
        let up1 = unimplemented!("TransposedConv2d not supported in candle");  // bottleneck
        let x = Tensor::cat(&[&up1, &enc2_skip], 1)?;
        let x = self.Conv2d_3.forward(&x)?;
        let dec1 = x.relu()?;
        let up2 = unimplemented!("TransposedConv2d not supported in candle");  // dec1
        let x = Tensor::cat(&[&up2, &enc1_skip], 1)?;
        let x = self.Conv2d_4.forward(&x)?;
        let x = x.relu()?;
        let x = self.Conv2d_5.forward(&x)?;
        Ok(x)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn setup() -> (candle_core::Device, VarMap, candle_nn::VarBuilder<'static>) {
        let dev = candle_core::Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &dev);
        (dev, varmap, vb)
    }

    #[test]
    fn test_forward() -> candle_core::Result<()> {
        let (dev, _varmap, vb) = setup();
        let model = UNet::new(vb)?;
        let x = candle_core::Tensor::randn(0f32, 1.0, &[1, 1, 64, 64], &dev)?;
        let output = model.forward(&x)?;
        assert_eq!(output.dims(), &[1, 1, 64, 64]);
        Ok(())
    }

    #[test]
    fn test_save_load() -> candle_core::Result<()> {
        let (dev, varmap, vb) = setup();
        let model = UNet::new(vb)?;
        let x = candle_core::Tensor::randn(0f32, 1.0, &[1, 1, 64, 64], &dev)?;
        let output_before = model.forward(&x)?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("UNet.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = UNet::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward(&x)?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}

