use candle_core::{Result, Tensor};
use candle_nn::Module;

pub struct Attention {
    Linear_0: candle_nn::Linear,
    Linear_1: candle_nn::Linear,
    Linear_2: candle_nn::Linear,
    Linear_3: candle_nn::Linear,
}

impl Attention {
    pub fn new(
        Linear_0: candle_nn::Linear,
        Linear_1: candle_nn::Linear,
        Linear_2: candle_nn::Linear,
        Linear_3: candle_nn::Linear,
    ) -> Self {
        Self {
            Linear_0,
            Linear_1,
            Linear_2,
            Linear_3,
        }
    }

    pub fn forward(&self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor
    ) -> Result<Tensor> {
        let q_proj = self.Linear_0.forward(&query)?;
        let k_proj = self.Linear_1.forward(&key)?;
        let v_proj = self.Linear_2.forward(&value)?;
        let x = q_proj.matmul(&k_proj)?;
        let attn_weights = candle_nn::ops::softmax(&x, candle_core::D::Minus1)?;
        let x = attn_weights.matmul(&v_proj)?;
        let x = self.Linear_3.forward(&x)?;
        Ok(x)
    }
}
