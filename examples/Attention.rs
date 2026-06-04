use candle_core::{ModuleT, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub struct Attention {
    Linear_0: candle_nn::Linear,
    Linear_1: candle_nn::Linear,
    Linear_2: candle_nn::Linear,
    Linear_3: candle_nn::Linear,
}

impl Attention {
    /// Create model with auto-generated weight scope names.
    /// Use [Attention::with_scopes] for custom scope names.
    pub fn new(weights: VarBuilder) -> Result<Self> {
        Self::with_scopes(weights,
        "Linear_0",
        "Linear_1",
        "Linear_2",
        "Linear_3",
        )
    }

    /// Create model with custom weight-loading scope names.
    pub fn with_scopes(
        weights: VarBuilder,
        linear_0: &str,
        linear_1: &str,
        linear_2: &str,
        linear_3: &str,
    ) -> Result<Self> {
        let Linear_0 = candle_nn::linear(64, 64, weights.pp(linear_0))?;
        let Linear_1 = candle_nn::linear(64, 64, weights.pp(linear_1))?;
        let Linear_2 = candle_nn::linear(64, 64, weights.pp(linear_2))?;
        let Linear_3 = candle_nn::linear(64, 64, weights.pp(linear_3))?;
        Ok(Self {
            Linear_0,
            Linear_1,
            Linear_2,
            Linear_3,
        })
    }

    pub fn forward(&self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor
    ) -> Result<Tensor> {
        let q_proj = self.Linear_0.forward(&query)?;
        let k_proj = self.Linear_1.forward(&key)?;
        let v_proj = self.Linear_2.forward(&value)?;
        let x = q_proj.matmul(&k_proj.t()?)?;
        let attn_weights = candle_nn::ops::softmax(&x, candle_core::D::Minus1)?;
        let x = attn_weights.matmul(&v_proj)?;
        let x = self.Linear_3.forward(&x)?;
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
        let model = Attention::new(vb)?;
        let query = candle_core::Tensor::randn(0f32, 1.0, &[1, 512, 64], &dev)?;
        let key = candle_core::Tensor::randn(0f32, 1.0, &[1, 512, 64], &dev)?;
        let value = candle_core::Tensor::randn(0f32, 1.0, &[1, 512, 64], &dev)?;
        let output = model.forward(&query, &key, &value)?;
        assert_eq!(output.dims(), &[1, 512, 64]);
        Ok(())
    }

    #[test]
    fn test_save_load() -> candle_core::Result<()> {
        let (dev, varmap, vb) = setup();
        let model = Attention::new(vb)?;
        let query = candle_core::Tensor::randn(0f32, 1.0, &[1, 512, 64], &dev)?;
        let key = candle_core::Tensor::randn(0f32, 1.0, &[1, 512, 64], &dev)?;
        let value = candle_core::Tensor::randn(0f32, 1.0, &[1, 512, 64], &dev)?;
        let output_before = model.forward(&query, &key, &value)?;

        let dir = std::env::temp_dir().join("mlmd-e2e");
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("Attention.safetensors");
        let _ = std::fs::remove_file(&path);
        varmap.save(&path)?;

        let mut varmap2 = VarMap::new();
        let vb2 = candle_nn::VarBuilder::from_varmap(&varmap2, candle_core::DType::F32, &dev);
        let model2 = Attention::new(vb2)?;
        varmap2.load(&path)?;
        let output_after = model2.forward(&query, &key, &value)?;

        let diff = (output_before - &output_after)?.abs()?.sum_all()?;
        assert!(diff.to_vec0::<f32>()? < 1e-5);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}

