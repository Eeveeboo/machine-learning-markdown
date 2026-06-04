use candle_core::{ModuleT, Result, Tensor};
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
    /// Create model with auto-generated weight scope names.
    /// Use [ResNet50::with_scopes] for custom scope names.
    pub fn new(weights: VarBuilder) -> Result<Self> {
        Self::with_scopes(weights,
        "Conv2d_0",
        "BatchNorm_0",
        "Conv2d_1",
        "BatchNorm_1",
        "Conv2d_2",
        "BatchNorm_2",
        "Conv2d_3",
        "BatchNorm_3",
        )
    }

    /// Create model with custom weight-loading scope names.
    pub fn with_scopes(
        weights: VarBuilder,
        conv2d_0: &str,
        batchNorm_0: &str,
        conv2d_1: &str,
        batchNorm_1: &str,
        conv2d_2: &str,
        batchNorm_2: &str,
        conv2d_3: &str,
        batchNorm_3: &str,
    ) -> Result<Self> {
        let Conv2d_0 = candle_nn::conv2d(256, 64, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_0))?;
        let BatchNorm_0 = candle_nn::batch_norm(64, 1e-5, weights.pp(batchNorm_0))?;
        let Conv2d_1 = candle_nn::conv2d(64, 64, 3, candle_nn::Conv2dConfig { stride: 1, padding: 1, ..Default::default() }, weights.pp(conv2d_1))?;
        let BatchNorm_1 = candle_nn::batch_norm(64, 1e-5, weights.pp(batchNorm_1))?;
        let Conv2d_2 = candle_nn::conv2d(64, 256, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_2))?;
        let BatchNorm_2 = candle_nn::batch_norm(256, 1e-5, weights.pp(batchNorm_2))?;
        let Conv2d_3 = candle_nn::conv2d(256, 256, 1, candle_nn::Conv2dConfig { stride: 1, padding: 0, ..Default::default() }, weights.pp(conv2d_3))?;
        let BatchNorm_3 = candle_nn::batch_norm(256, 1e-5, weights.pp(batchNorm_3))?;
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

    pub fn forward(&self,
        skip: &Tensor
    ) -> Result<Tensor> {
        let x = self.Conv2d_0.forward(&skip)?;
        let x = self.BatchNorm_0.forward_t(&x, false)?;
        let x = x.relu()?;
        let x = self.Conv2d_1.forward(&x)?;
        let x = self.BatchNorm_1.forward_t(&x, false)?;
        let x = x.relu()?;
        let x = self.Conv2d_2.forward(&x)?;
        let main_out = self.BatchNorm_2.forward_t(&x, false)?;
        let x = self.Conv2d_3.forward(&skip)?;
        let skip_out = self.BatchNorm_3.forward_t(&x, false)?;
        let x = (&main_out + &skip_out)?;
        let x = x.relu()?;
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
        let model = ResNet50::new(vb)?;
        let skip = candle_core::Tensor::randn(0f32, 1.0, &[1, 256, 56, 56], &dev)?;
        let output = model.forward(&skip)?;
        assert_eq!(output.dims(), &[1, 256, 56, 56]);
        Ok(())
    }

    #[test]
    fn test_save_load() -> candle_core::Result<()> {
        let (dev, varmap, vb) = setup();
        let model = ResNet50::new(vb)?;
        let skip = candle_core::Tensor::randn(0f32, 1.0, &[1, 256, 56, 56], &dev)?;
        let output_before = model.forward(&skip)?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("ResNet50.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = ResNet50::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward(&skip)?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}

