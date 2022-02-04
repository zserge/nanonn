use crate::activation::*;
use crate::layer::Layer;
use rand::Rng;

pub struct Dense {
    input_count: usize,
    output_count: usize,
    weights: Vec<f64>,
    activation: Box<dyn Activation>,
    outputs: Vec<f64>,
    errors: Vec<f64>,
}

impl Dense {
    pub fn new(
        inputs: usize,
        units: usize,
        activation: Box<dyn Activation>,
        o_weights: Option<Vec<f64>>,
    ) -> Dense {
        let weights = o_weights.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            let mut w = vec![0.0; units * (inputs + 1)];
            for w in w.iter_mut() {
                *w = rng.gen::<f64>() * 2.0 - 1.0;
            }
            w
        });
        // bounds check, because unsafe
        assert!(
            weights.len() == units * (inputs + 1),
            "Invalid weights for Dense layer"
        );

        Dense {
            input_count: inputs,
            output_count: units,
            weights,
            activation,
            outputs: vec![0.0; units],
            errors: vec![0.0; inputs],
        }
    }
}

impl Layer for Dense {
    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_count, "Invalid input size");
        unsafe {
            let n = self.input_count + 1;
            let mut outputs = vec![0.0; self.output_count];
            for i in 0..self.output_count {
                let mut sum = 0.0;
                for j in 0..self.input_count {
                    sum += input.get_unchecked(j) * self.weights.get_unchecked(i * n + j);
                }
                *outputs.get_unchecked_mut(i) = self
                    .activation
                    .activate(sum + self.weights.get_unchecked(i * n + n - 1));
            }
            self.outputs = outputs.clone();
            outputs
        }
    }

    fn backward(&mut self, input: &[f64], e: &[f64], rate: f64) -> Vec<f64> {
        assert_eq!(input.len(), self.input_count, "Invalid input size");
        assert_eq!(e.len(), self.output_count, "Invalid error size");
        unsafe {
            let n = self.input_count + 1;
            let mut errors_out = vec![0.0; self.input_count];
            for j in 0..self.input_count {
                let mut sum = 0.0;
                for i in 0..self.output_count {
                    sum += e.get_unchecked(i)
                        * self.activation.derivative(*self.outputs.get_unchecked(i))
                        * self.weights.get_unchecked(i * n + j);
                }
                *errors_out.get_unchecked_mut(j) = sum;
            }
            for i in 0..self.output_count {
                for j in 0..self.input_count {
                    *self.weights.get_unchecked_mut(i * n + j) += rate
                        * e.get_unchecked(i)
                        * self.activation.derivative(*self.outputs.get_unchecked(i))
                        * input.get_unchecked(j);
                }
                *self.weights.get_unchecked_mut(i * n + n - 1) += rate
                    * e.get_unchecked(i)
                    * self.activation.derivative(*self.outputs.get_unchecked(i));
            }
            self.errors = errors_out.clone();
            errors_out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_eq_approx {
        ($a:expr, $b:expr, $precision:expr) => {
            assert!(($a - $b).abs() < $precision, "{} != {}", $a, $b)
        };
    }

    #[test]
    fn test_forward() {
        let l_weights = vec![1.74481176, -0.7612069, 0.3190391, -0.24937038];
        let mut l = Dense::new(3, 1, Box::new(Sigmoid), Some(l_weights));

        let x1 = [1.62434536, -0.52817175, 0.86540763];
        let y1 = 0.96313579;
        let z1 = l.forward(&x1);
        assert_eq_approx!(y1, z1[0], 0.001);

        let x2 = [-0.61175641, -1.07296862, -2.3015387];
        let y2 = 0.22542973;
        let z2 = l.forward(&x2);
        assert_eq_approx!(y2, z2[0], 0.001);
    }
}
