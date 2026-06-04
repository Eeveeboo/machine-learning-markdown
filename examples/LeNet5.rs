use candle_core::{ModuleT, Result, Tensor};
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
        let x = x.max_pool2d(2)?;
        let x = self.Conv2d_1.forward(&x)?;
        let x = x.relu()?;
        let x = x.max_pool2d(2)?;
        let x = x.flatten_from(1)?;
        let x = self.Linear_0.forward(&x)?;
        let x = x.relu()?;
        let x = self.Linear_1.forward(&x)?;
        let x = x.relu()?;
        let x = self.Linear_2.forward(&x)?;
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
        let model = LeNet5::new(vb)?;
        let x = candle_core::Tensor::randn(0f32, 1.0, &[1, 1, 28, 28], &dev)?;
        let output = model.forward(&x)?;
        assert_eq!(output.dims(), &[1, 10]);
        Ok(())
    }

    #[test]
    fn test_save_load() -> candle_core::Result<()> {
        let (dev, varmap, vb) = setup();
        let model = LeNet5::new(vb)?;
        let x = candle_core::Tensor::randn(0f32, 1.0, &[1, 1, 28, 28], &dev)?;
        let output_before = model.forward(&x)?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("LeNet5.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = LeNet5::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward(&x)?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}

