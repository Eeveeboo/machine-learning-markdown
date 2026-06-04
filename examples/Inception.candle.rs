use candle_core::{ModuleT, Result, Tensor};
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
    /// Create model with auto-generated weight scope names.
    /// Use [Inception::with_scopes] for custom scope names.
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
        let Conv2d_0 = candle_nn::conv2d(192, 64, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_0))?;
        let Conv2d_1 = candle_nn::conv2d(192, 96, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_1))?;
        let Conv2d_2 = candle_nn::conv2d(96, 128, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_2))?;
        let Conv2d_3 = candle_nn::conv2d(192, 16, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_3))?;
        let Conv2d_4 = candle_nn::conv2d(16, 32, 5, candle_nn::Conv2dConfig { stride: 1, padding: 2, ..Default::default() }, weights.pp(conv2d_4))?;
        let Conv2d_5 = candle_nn::conv2d(192, 32, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_5))?;
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
        b4: &Tensor
    ) -> Result<Tensor> {
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
        let x = b4.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?.max_pool2d_with_stride(3, 1)?;
        let x = self.Conv2d_5.forward(&x)?;
        let out4 = x.relu()?;
        let x = Tensor::cat(&[&out1, &out2, &out3, &out4], 1)?;
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
        let model = Inception::new(vb)?;
        let b4 = candle_core::Tensor::randn(0f32, 1.0, &[1, 192, 28, 28], &dev)?;
        let output = model.forward(&b4)?;
        assert_eq!(output.dims(), &[1, 256, 28, 28]);
        Ok(())
    }

    #[test]
    fn test_save_load() -> candle_core::Result<()> {
        let (dev, varmap, vb) = setup();
        let model = Inception::new(vb)?;
        let b4 = candle_core::Tensor::randn(0f32, 1.0, &[1, 192, 28, 28], &dev)?;
        let output_before = model.forward(&b4)?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("Inception.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = Inception::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward(&b4)?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
