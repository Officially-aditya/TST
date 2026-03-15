pub struct ModelConfig {
    pub bias_scale: f32,
    pub bias_clamp_min: f32,
    pub bias_clamp_max: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            bias_scale: 1.0,
            bias_clamp_min: -1.5,
            bias_clamp_max: 1.5,
        }
    }
}

pub fn compute_bias(frequency: u32, decay_score: f32, config: &ModelConfig) -> f32 {
    // Guard: NaN or Inf decay_score produces undefined bias — return 0.0 (no influence).
    if !decay_score.is_finite() {
        return 0.0;
    }
    let raw_bias = ((1 + frequency) as f32).ln() * decay_score * config.bias_scale;
    raw_bias.clamp(config.bias_clamp_min, config.bias_clamp_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bias_computation() {
        let config = ModelConfig::default();
        let bias = compute_bias(10, 0.9, &config);
        
        // ln(11) * 0.9 ≈ 2.39 * 0.9 = 2.15 >> clamped to 1.5
        assert_eq!(bias, 1.5);
        
        let low_bias = compute_bias(1, 0.5, &config);
        // ln(2) * 0.5 ≈ 0.69 * 0.5 = 0.345
        assert!((low_bias - 0.3465736).abs() < 0.001);
    }
}
