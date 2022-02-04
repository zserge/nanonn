pub trait Layer {
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
    fn get_weights(&self) -> Vec<f64>;
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;
    fn backward(&mut self, input: &[f64], e: &[f64], rate: f64) -> Vec<f64>;
}
