use crate::layer::Layer;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    errors: Vec<f64>,
    inputs: Vec<Vec<f64>>,
    input_count: usize,
    output_count: usize,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Network {
        assert!(layers.len() > 0, "Network must have at least one layer");
        let input_count = layers.first().unwrap().input_count();
        let output_count = layers.last().unwrap().output_count();
        let errors = vec![0.0; layers.last().unwrap().output_count()];
        let inputs = (0..layers.len())
            .map(|_| vec![0.0; layers[0].input_count()])
            .collect();
        // check that the layers are compatible, for unsafe
        for (a, b) in layers.iter().zip(layers.iter().skip(1)) {
            assert_eq!(
                a.output_count(),
                b.input_count(),
                "Invalid layer configuration"
            );
        }
        Network {
            layers,
            errors,
            inputs,
            input_count,
            output_count,
        }
    }

    pub fn predict(&mut self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.input_count, "Invalid input size");
        let mut inputs = x.to_vec();
        for layer in self.layers.iter_mut() {
            let outputs = layer.forward(&inputs);
            inputs = outputs;
        }
        inputs
    }

    pub fn train(&mut self, x: &[f64], y: &[f64], rate: f64) -> f64 {
        assert_eq!(x.len(), self.input_count, "Invalid input size");
        assert_eq!(y.len(), self.output_count, "Invalid expected output size");
        unsafe {
            let mut inputs = x.to_vec();
            for (i, layer) in self.layers.iter_mut().enumerate() {
                *self.inputs.get_unchecked_mut(i) = inputs.clone();
                let outputs = layer.forward(&inputs);
                inputs = outputs;
            }
            let mut e = 0.0;
            for (i, yi) in y.iter().enumerate() {
                *self.errors.get_unchecked_mut(i) = yi - inputs.get_unchecked(i);
                let ei = self.errors.get_unchecked(i);
                e += ei * ei;
            }
            let mut errors = self.errors.clone();
            for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                errors = layer.backward(self.inputs.get_unchecked(i), &errors, rate);
            }
            e / y.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;
    use crate::activation::Sigmoid;
    use crate::dense::Dense;

    macro_rules! assert_eq_vec_approx {
        ($a:expr, $b:expr, $precision:expr) => {
            assert!(
                $a.iter()
                    .zip($b.iter())
                    .all(|(a, b)| (a - b).abs() < $precision),
                "a: {:?}, b: {:?}",
                $a,
                $b
            );
        };
    }

    macro_rules! assert_eq_approx {
        ($a:expr, $b:expr, $precision:expr) => {
            assert!(($a - $b).abs() < $precision, "{} != {}", $a, $b)
        };
    }

    #[test]

    fn test_weights() {
        let l1_weights = vec![0.15, 0.2, 0.35, 0.25, 0.3, 0.35];
        let l2_weights = vec![0.4, 0.45, 0.6, 0.5, 0.55, 0.6];

        let l1 = Box::new(Dense::new(2, 2, Box::new(Sigmoid), Some(l1_weights)));
        let l2 = Box::new(Dense::new(2, 2, Box::new(Sigmoid), Some(l2_weights)));

        let mut n = Network::new(vec![l1, l2]);

        // Ensure forward propagation works for both layers
        let z = n.predict(&[0.05, 0.1]);
        assert_eq_approx!(0.75136507, z[0], 0.0001);
        assert_eq_approx!(0.772928465, z[1], 0.0001);

        // Ensure that squared error is calculated correctly (use rate=0 to avoid training)
        let e = n.train(&[0.05, 0.1], &[0.01, 0.99], 0.0);
        assert_eq_approx!(0.298371109, e, 0.0001);

        // Backpropagation with rate 0.5
        n.train(&[0.05, 0.1], &[0.01, 0.99], 0.5);

        // Check weights
        let l2_weights_expected = vec![
            0.35891648,
            0.408666186,
            0.530751,
            0.511301270,
            0.561370121,
            0.619049,
        ];
        let l1_weights_expected = vec![
            0.149780716,
            0.19956143,
            0.345614,
            0.24975114,
            0.29950229,
            0.345023,
        ];

        assert_eq_vec_approx!(l2_weights_expected, n.layers[1].get_weights(), 0.001);
        assert_eq_vec_approx!(l1_weights_expected, n.layers[0].get_weights(), 0.001);
    }

    #[test]
    fn test_layer_or() {
        let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = [[0.0], [1.0], [1.0], [1.0]];
        let mut e = [0.0; 1];
        let mut l = Box::new(Dense::new(2, 1, Box::new(Sigmoid), None));
        for _ in 0..1000 {
            for (i, x) in x.iter().enumerate() {
                let z = l.forward(x);
                e[0] = y[i][0] - z[0];
                l.backward(x, &e, 1.0);
            }
        }
        for (i, x) in x.iter().enumerate() {
            let z = l.forward(x);
            assert_eq_approx!(y[i][0], z[0], 0.1);
        }
    }

    #[test]
    fn test_network_xor() {
        let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = [[0.0], [1.0], [1.0], [0.0]];

        let mut n = Network::new(vec![
            Box::new(Dense::new(2, 4, Box::new(Sigmoid), None)),
            Box::new(Dense::new(4, 1, Box::new(Sigmoid), None)),
        ]);
        let mut successful_train = false;
        for _ in 0..10000 {
            let mut e = 0.0;
            for (xi, yi) in x.iter().zip(y.iter()) {
                e += n.train(xi, yi, 1.0);
            }
            if e < 0.02 {
                successful_train = true;
                break;
            }
        }
        assert!(successful_train, "Failed to train the model");
    }

    #[test]
    fn test_network_sinc() {
        let sinc = |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x };
        let mut n = Network::new(vec![
            Box::new(Dense::new(1, 5, Box::new(Sigmoid), None)),
            Box::new(Dense::new(5, 10, Box::new(Sigmoid), None)),
            Box::new(Dense::new(10, 1, Box::new(Sigmoid), None)),
        ]);
        let mut rng = rand::thread_rng();
        let mut successful_train = false;
        for _ in 0..1000 {
            let mut e = 0.0;
            for _ in 0..100 {
                let x = rng.gen::<f64>() * 10.0 - 5.0;
                e += n.train(&[x], &[sinc(x)], 0.5) / 100.0;
            }
            if e < 0.01 {
                successful_train = true;
                break;
            }
        }
        assert!(successful_train, "Failed to train the model");
    }
}
