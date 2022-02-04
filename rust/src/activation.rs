pub trait Activation {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }
}

pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub struct LeakyReLU;

impl Activation for LeakyReLU {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.01 * x
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }
}

pub struct SoftPlus;

impl Activation for SoftPlus {
    fn activate(&self, x: f64) -> f64 {
        (1.0 + (-x).exp()).ln()
    }

    fn derivative(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

pub struct Linear;

impl Activation for Linear {
    fn activate(&self, x: f64) -> f64 {
        x
    }

    fn derivative(&self, _x: f64) -> f64 {
        1.0
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
    fn test_sigmoid() {
        let mut y = Sigmoid.activate(0.0);
        assert_eq!(y, 0.5);
        y = Sigmoid.activate(2.0);
        assert_eq_approx!(y, 0.8807970779778823, 0.0001);
    }
}
