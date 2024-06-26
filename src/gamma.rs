use statrs::function::gamma::{gamma, gamma_lr, gamma_ur};

pub fn inverse_gamma_lr(a: f64, p: f64, n_iter: usize) -> f64 {
    // this algorithm is taken from https://dl.acm.org/doi/pdf/10.1145/22721.23109

    // get an estimate for x0 to start newton iterations.
    let q = 1.0 - p;

    if (1.0 - 1.0e-8..=1.0 + 1.0e-8).contains(&a) {
        return -q.ln();
    }

    let gamma_a = gamma(a);
    let b = q * gamma_a;
    let c = 0.577_215_664_901_532_9;

    let mut x0 = 0.5;
    if a < 1.0 {
        if b > 0.6 || (b >= 0.45 && a >= 0.3) {
            let u = if b * q > 10e-8 {
                (p * gamma(a + 1.0)).powf(a.recip())
            } else {
                (-q / a - c).exp()
            };
            x0 = u / (1.0 - u / (a + 1.0));
        } else if a < 0.3 && (0.35..=0.6).contains(&b) {
            let t = (-c - b).exp();
            let u = t * t.exp();
            x0 = t * u.exp();
        } else if (0.15..=0.35).contains(&b) || ((0.15..0.45).contains(&b) && a >= 0.3) {
            let y = -b.ln();
            let u = y - (1.0 - a) * y.ln();
            x0 = y - (1.0 - a) * y.ln() - (1.0 + (1.0 - a) / (1.0 + u)).ln();
        } else if 0.01 < b && b < 0.15 {
            let y = -b.ln();
            let u = y - (1.0 - a) * y.ln();
            x0 = y
                - (1.0 - a) * u.ln()
                - ((u * u + 2.0 * (3.0 - a) * u + (2.0 - a) * (3.0 - a))
                    / (u * u + (5.0 - a) * u + 2.0))
                    .ln();
        } else if b <= 0.01 {
            let y = -b.ln();
            let c1 = (a - 1.0) * y.ln();
            let c2 = (a - 1.0) * (1.0 + c1);
            let c3 = (a - 1.0) * (-0.5 * c1 * c1 + (a - 2.0) * c1 + (3.0 * a - 5.0) * 0.5);
            let c4 = (a - 1.0)
                * (1.0 / 3.0 * c1 * c1 * c1 - (3.0 * a - 5.0) * 0.5 * c1 * c1
                    + (a * a - 6.0 * a + 7.0) * c1
                    + (11.0 * a * a - 46.0 * a + 47.0) / 6.0);
            let c5 = (a - 1.0)
                * (-0.25 * c1 * c1 * c1 * c1
                    + (11.0 * a - 7.0) / 6.0 * c1 * c1 * c1
                    + (-3.0 * a * a - 13.0) * c1 * c1
                    + (2.0 * a * a * a - 25.0 * a * a + 72.0 * a - 61.0) * 0.5 * c1
                    + (25.0 * a * a * a - 195.0 * a * a + 477.0 * a - 379.0) / 12.0);
            x0 = y + c1 + c2 / (y) + c3 / (y * y) + c4 / (y * y * y) + c5 / (y * y * y * y);

            if b <= 1.0e-28 {
                return x0;
            }
        }
    } else {
        let pref;
        let tau;
        if p < 0.5 {
            pref = -1.0;
            tau = p;
        } else {
            pref = 1.0;
            tau = q;
        }
        let t = (-2.0 * tau.ln()).sqrt();

        let a_0 = 3.31125922108741;
        let a_1 = 11.6616720288968;
        let a_2 = 4.28342155967104;
        let a_3 = 0.213623493715853;

        let b_1 = 6.61053765625462;
        let b_2 = 6.40691597760039;
        let b_3 = 1.27364489782223;
        let b_4 = 3.611_708_101_884_203e-2;

        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;

        let numerator = a_0 + a_1 * t + a_2 * t2 + a_3 * t3;
        let denominator = 1.0 + b_1 * t + b_2 * t2 + b_3 * t3 + b_4 * t4;

        let s = pref * (t - numerator / denominator);
        let s2 = s * s;
        let s3 = s * s2;
        let s4 = s * s3;
        let s5 = s * s4;

        let a_sqrt = a.sqrt();

        let w = a + s * a_sqrt + (s2 - 1.0) / 3.0 + (s3 - 7.0 * s) / (36.0 * a_sqrt)
            - (3.0 * s4 + 7.0 * s2 - 16.0) / (810.0 * a)
            + (9.0 * s5 + 256.0 * s3 - 433.0 * s) / (38880.0 * a * a_sqrt);

        if a >= 500.0 && (1.0 - w / a).abs() < 1.0e-6 {
            return w;
        } else if p > 0.5 {
            if w < 3.0 * a {
                x0 = w;
            } else {
                let d = 2f64.max(a * (a - 1.0));
                if b > 10f64.powf(-d) {
                    let u = -b.ln() + (a - 1.0) * w.ln() - (1.0 + (1.0 - a) / (1.0 + w)).ln();
                    x0 = -b.ln() + (a - 1.0) * u.ln() - (1.0 + (1.0 - a) / (1.0 + u)).ln();
                } else {
                    let y = -b.ln();
                    let c1 = (a - 1.0) * y.ln();
                    let c2 = (a - 1.0) * (1.0 + c1);
                    let c3 = (a - 1.0) * (-0.5 * c1 * c1 + (a - 2.0) * c1 + (3.0 * a - 5.0) * 0.5);
                    let c4 = (a - 1.0)
                        * (1.0 / 3.0 * c1 * c1 * c1 - (3.0 * a - 5.0) * 0.5 * c1 * c1
                            + (a * a - 6.0 * a + 7.0) * c1
                            + (11.0 * a * a - 46.0 * a + 47.0) / 6.0);
                    let c5 = (a - 1.0)
                        * (-0.25 * c1 * c1 * c1 * c1
                            + (11.0 * a - 7.0) / 6.0 * c1 * c1 * c1
                            + (-3.0 * a * a - 13.0) * c1 * c1
                            + (2.0 * a * a * a - 25.0 * a * a + 72.0 * a - 61.0) * 0.5 * c1
                            + (25.0 * a * a * a - 195.0 * a * a + 477.0 * a - 379.0) / 12.0);
                    x0 = y + c1 + c2 / (y) + c3 / (y * y) + c4 / (y * y * y) + c5 / (y * y * y * y);
                }
            }
        } else {
            // this part is heavily simplified from the paper, if any issues occur this estimate
            // will need more refinement.
            let v = (p * gamma(a + 1.0)).ln();
            x0 = ((v + w) / a).exp();
        }
    }

    // start iteration
    let mut x_n = x0;
    for _ in 0..n_iter {
        let r = x_n.powf(a - 1.0) * (-x_n).exp() / gamma_a;
        if x_n <= 0. {
            x_n = 1.0e-16;
        }
        let t_n = if p <= 0.5 {
            (gamma_lr(a, x_n) - p) / r
        } else {
            -(gamma_ur(a, x_n) - q) / r
        };
        let w_n = (a - 1.0 - x_n) / 2.0;

        let h_n = if t_n.abs() <= 0.1 && (w_n * t_n).abs() <= 0.1 {
            t_n + w_n * t_n * t_n
        } else {
            t_n
        };

        x_n -= h_n;
    }

    x_n
}

#[cfg(test)]
mod tests {
    use statrs::function::gamma::gamma_lr;

    use crate::assert_approx_eq;

    use super::inverse_gamma_lr;

    const NITER_FOR_TEST: usize = 50;
    const TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn test_1() {
        let omega = 1.;

        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let inverse_lower_gamma = inverse_gamma_lr(omega, p, NITER_FOR_TEST);
            assert_approx_eq(gamma_lr(omega, inverse_lower_gamma), p, TOLERANCE);
        }
    }

    #[test]
    fn test_half() {
        let omega = 0.5;
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let inverse_lower_gamma = inverse_gamma_lr(omega, p, NITER_FOR_TEST);
            assert_approx_eq(gamma_lr(omega, inverse_lower_gamma), p, TOLERANCE);
        }
    }

    #[test]
    fn test_2() {
        let omega = 2.0;
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let inverse_lower_gamma = inverse_gamma_lr(omega, p, NITER_FOR_TEST);
            assert_approx_eq(gamma_lr(omega, inverse_lower_gamma), p, TOLERANCE);
        }
    }

    #[test]
    fn test_10() {
        let omega = 10.0;
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let inverse_lower_gamma = inverse_gamma_lr(omega, p, NITER_FOR_TEST);
            assert_approx_eq(gamma_lr(omega, inverse_lower_gamma), p, TOLERANCE);
        }
    }
}
