use smallvec::SmallVec;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use crate::{TropicalSamplingSettings, float::MomTropFloat};

#[derive(Debug, Clone, Default)]
/// square symmetric matrix for use in the tropical sampling algorithm
pub struct SquareMatrix<T> {
    data: SmallVec<[T; 36]>, // this allows us to store the L matrix on the stack for 6 loops and below
    dim: usize,
}

#[derive(Clone, Copy, Debug)]
/// Error type for when problems occur in the matrix algorithms
pub enum MatrixError {
    /// Used when the determinant of L is zero.
    ZeroDet,
    /// Used when the stability condition is not met.
    Unstable,
}

impl<T> Index<(usize, usize)> for SquareMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.dim + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for SquareMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.dim + index.1]
    }
}

// silly clippy
#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a, T: MomTropFloat> Mul<&'a SquareMatrix<T>> for &'a SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = self.new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                for k in 0..self.dim {
                    result[(row, col)] += &self[(row, k)].ref_mul(&rhs[(k, col)]);
                }
            }
        }

        result
    }
}

impl<T: MomTropFloat> Mul<T> for SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self.new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                result[(row, col)] = self[(row, col)].ref_mul(&rhs);
            }
        }

        result
    }
}

impl<'a, T: MomTropFloat> Add<&'a SquareMatrix<T>> for &'a SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                result[(row, col)] = self[(row, col)].ref_add(&rhs[(row, col)]);
            }
        }

        result
    }
}

impl<'a, T: MomTropFloat> Sub<&'a SquareMatrix<T>> for &'a SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self.new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                result[(row, col)] = self[(row, col)].ref_sub(&rhs[(row, col)]);
            }
        }

        result
    }
}

impl<T: MomTropFloat> SquareMatrix<T> {
    #[must_use]
    pub fn new_zeros(&self, dim: usize) -> Self {
        Self {
            data: SmallVec::from_elem(self.data[0].zero(), dim * dim),
            dim,
        }
    }

    pub fn new_zeros_from_num(builder: &T, dim: usize) -> Self {
        Self {
            data: SmallVec::from_elem(builder.zero(), dim * dim),
            dim,
        }
    }

    /// Performs operations on a matrix for tropical sampling
    /// # Errors
    /// Returns an error if the matrix is not invertible
    pub fn decompose_for_tropical(
        &self,
        settings: &TropicalSamplingSettings,
    ) -> Result<DecompositionResult<T>, MatrixError> {
        if settings.print_debug_info {
            println!("starting cholesky")
        }
        let const_builder = &self.data[0];

        // start cholesky decomposition
        let mut q = self.new_zeros(self.dim);

        for i in 0..self.dim {
            let mut diagonal_entry_squared = self[(i, i)].clone();
            for j in 0..i {
                diagonal_entry_squared -= &q[(i, j)].ref_mul(&q[(i, j)]);
            }

            let diagonal_entry = diagonal_entry_squared.sqrt();
            q[(i, i)] = diagonal_entry.clone();

            for j in i + 1..self.dim {
                let mut entry = self[(i, j)].clone();
                for k in 0..i {
                    entry -= &q[(i, k)].ref_mul(&q[(j, k)]);
                }
                q[(j, i)] = entry / &diagonal_entry;
            }
        }
        // end cholesky decomposition

        // compute the determinant of Q and store the invesrses of the diagonal elements
        let mut det_q = const_builder.one();
        let mut inverse_diagonal_entries = SmallVec::<[T; 6]>::new();

        for i in 0..self.dim {
            let q_ii = q[(i, i)].clone();
            det_q *= &q_ii;
            inverse_diagonal_entries.push(q_ii.inv());
        }

        (0..self.dim).fold(const_builder.one(), |acc, i| acc * &q[(i, i)]);
        let determinant = det_q.ref_mul(&det_q);

        if det_q == const_builder.zero() {
            return Err(MatrixError::ZeroDet);
        }

        // the matrix N is defined through Q = D(I + N)
        let mut n_matrix = self.new_zeros(self.dim);
        for row in 1..self.dim {
            let inverse_diagonal_element = &inverse_diagonal_entries[row];
            for col in 0..row {
                n_matrix[(row, col)] = inverse_diagonal_element.ref_mul(&q[(row, col)]);
            }
        }

        let max_non_zero_power_of_n = self.dim - 1;

        // this algorithm is unoptizmied, will optimize later if this works
        let mut powers_of_n = SmallVec::<[SquareMatrix<T>; 5]>::new();
        powers_of_n.push(n_matrix);

        for _ in 1..max_non_zero_power_of_n {
            let last_power_of_n = powers_of_n
                .last()
                .unwrap_or_else(|| unreachable!("Never empty due to push before"));
            let first_power_of_n = powers_of_n
                .first()
                .unwrap_or_else(|| unreachable!("Never empty due to push before"));
            powers_of_n.push(last_power_of_n * first_power_of_n);
        }

        let n_sum =
            powers_of_n
                .iter()
                .enumerate()
                .fold(self.new_zeros(self.dim), |acc, (i, mat)| {
                    if i % 2 == 0 { &acc - mat } else { &acc + mat }
                });

        let mut inverse_q = n_sum;
        for row in 0..self.dim {
            inverse_q[(row, row)] += &const_builder.one();
            for col in 0..self.dim {
                inverse_q[(row, col)] *= &inverse_diagonal_entries[col];
            }
        }

        let mut q_transposed_inverse = self.new_zeros(self.dim);
        let mut q_transposed = self.new_zeros(self.dim);
        for row in 0..self.dim {
            for col in 0..self.dim {
                q_transposed_inverse[(row, col)] = inverse_q[(col, row)].clone();
                q_transposed[(row, col)] = q[(col, row)].clone();
            }
        }

        let inverse = &q_transposed_inverse * &inverse_q;

        if let Some(tolerance) = settings.matrix_stability_test {
            if settings.print_debug_info {
                println!("Performing matrix inversion quality");
            }

            let approx_idendity = &inverse * self;
            let true_identity = self.new_identity(self.dim);
            let zero = &approx_idendity - &true_identity;
            let error = zero.l21_norm();

            if settings.print_debug_info {
                println!("error: {:?}", error);
            }

            if error > error.from_f64(tolerance) {
                if settings.print_debug_info {
                    println!("Inversion unstable");
                }

                return Err(MatrixError::Unstable);
            }
        }

        Ok(DecompositionResult {
            determinant,
            inverse,
            q_transposed_inverse,
            q_transposed,
        })
    }

    fn new_identity(&self, dim: usize) -> Self {
        let mut res = self.new_zeros(dim);
        for i in 0..dim {
            res[(i, i)] = self.data[0].one();
        }
        res
    }

    fn l21_norm(&self) -> T {
        let mut res = self.data[0].zero();

        for j in 0..self.dim {
            let mut vec_norm = res.zero();
            for i in 0..self.dim {
                vec_norm += &self[(i, j)].ref_mul(&self[(i, j)]);
            }
            res += &vec_norm.sqrt();
        }

        res
    }

    pub fn zero(&self) -> T {
        self.data[0].zero()
    }
}

impl<T> SquareMatrix<T> {
    pub fn get_dim(&self) -> usize {
        self.dim
    }

    pub fn get_raw_data(self) -> SmallVec<[T; 36]> {
        self.data
    }
}

#[derive(Debug, Clone)]
/// Struct containing the result of the matrix algorithms. q is the Cholesky decomposition of l.
pub struct DecompositionResult<T: MomTropFloat> {
    pub determinant: T,
    pub inverse: SquareMatrix<T>,
    pub q_transposed: SquareMatrix<T>,
    pub q_transposed_inverse: SquareMatrix<T>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    const EPSILON: f64 = 1e-12;

    fn builder_matrix_f64() -> SquareMatrix<f64> {
        SquareMatrix {
            data: smallvec::SmallVec::from_elem(1.0, 1),
            dim: 1,
        }
    }

    #[test]
    fn test_decompose_for_tropical_2x2() {
        println!("starting test");
        let mut test_matrix = builder_matrix_f64().new_zeros(2);
        println!("build matrix");

        let settings: TropicalSamplingSettings = TropicalSamplingSettings {
            print_debug_info: true,
            ..Default::default()
        };

        test_matrix[(0, 0)] = 2.0;
        test_matrix[(1, 1)] = 4.0;
        test_matrix[(0, 1)] = 1.0;
        test_matrix[(1, 0)] = 1.0;

        let decomposition_result = test_matrix.decompose_for_tropical(&settings).unwrap();

        assert_approx_eq(&decomposition_result.determinant, &7.0, &EPSILON);

        let identity = &test_matrix * &decomposition_result.inverse;

        assert_approx_eq(&identity[(0, 0)], &1.0, &EPSILON);
        assert_approx_eq(&identity[(0, 1)], &0.0, &EPSILON);
        assert_approx_eq(&identity[(1, 0)], &0.0, &EPSILON);
        assert_approx_eq(&identity[(1, 1)], &1.0, &EPSILON);
    }

    // #[test]
    // fn test_decompose_for_tropical_f128_2x2() {
    //     let mut test_matrix = SquareMatrix::new_zeros(2);
    //     let settings = TropicalSamplingSettings::default();

    //     test_matrix[(0, 0)] = f128::new(2.0);
    //     test_matrix[(1, 1)] = f128::new(4.0);
    //     test_matrix[(0, 1)] = f128::new(1.0);
    //     test_matrix[(1, 0)] = f128::new(1.0);

    //     let decomposition_result = test_matrix.decompose_for_tropical(&settings).unwrap();

    //     assert_approx_eq(
    //         Into::<f64>::into(decomposition_result.determinant),
    //         7.0,
    //         EPSILON,
    //     );

    //     let identity = &test_matrix * &decomposition_result.inverse;

    //     assert_approx_eq(Into::<f64>::into(identity[(0, 0)]), 1.0, EPSILON);
    //     assert_approx_eq(Into::<f64>::into(identity[(0, 1)]), 0.0, EPSILON);
    //     assert_approx_eq(Into::<f64>::into(identity[(1, 0)]), 0.0, EPSILON);
    //     assert_approx_eq(Into::<f64>::into(identity[(1, 1)]), 1.0, EPSILON);
    // }

    #[test]
    fn test_decompose_for_tropical_4x4() {
        let mut wilson_matrix = builder_matrix_f64().new_zeros(4);
        let settings: TropicalSamplingSettings = TropicalSamplingSettings::default();

        wilson_matrix[(0, 0)] = 5.0;
        wilson_matrix[(1, 1)] = 10.0;
        wilson_matrix[(2, 2)] = 10.0;
        wilson_matrix[(3, 3)] = 10.0;
        wilson_matrix[(0, 1)] = 7.0;
        wilson_matrix[(1, 0)] = 7.0;
        wilson_matrix[(0, 2)] = 6.0;
        wilson_matrix[(2, 0)] = 6.0;
        wilson_matrix[(0, 3)] = 5.0;
        wilson_matrix[(3, 0)] = 5.0;
        wilson_matrix[(1, 2)] = 8.0;
        wilson_matrix[(2, 1)] = 8.0;
        wilson_matrix[(1, 3)] = 7.0;
        wilson_matrix[(3, 1)] = 7.0;
        wilson_matrix[(2, 3)] = 9.0;
        wilson_matrix[(3, 2)] = 9.0;

        let decomposition_result = wilson_matrix.decompose_for_tropical(&settings).unwrap();

        let identity = &wilson_matrix * &decomposition_result.inverse;

        for row in 0..4 {
            for col in 0..4 {
                assert_approx_eq(
                    &identity[(row, col)],
                    if row == col { &1.0 } else { &0.0 },
                    &EPSILON,
                );
            }
        }

        assert_approx_eq(&decomposition_result.determinant, &1.0, &EPSILON);
    }

    //  #[test]
    //  fn test_decompose_for_tropical_4x4_f128() {
    //      let mut wilson_matrix = SquareMatrix::new_zeros(4);
    //      let settings = TropicalSamplingSettings::default();

    //      wilson_matrix[(0, 0)] = f128::new(5.0);
    //      wilson_matrix[(1, 1)] = f128::new(10.);
    //      wilson_matrix[(2, 2)] = f128::new(10.);
    //      wilson_matrix[(3, 3)] = f128::new(10.);
    //      wilson_matrix[(0, 1)] = f128::new(7.0);
    //      wilson_matrix[(1, 0)] = f128::new(7.0);
    //      wilson_matrix[(0, 2)] = f128::new(6.0);
    //      wilson_matrix[(2, 0)] = f128::new(6.0);
    //      wilson_matrix[(0, 3)] = f128::new(5.0);
    //      wilson_matrix[(3, 0)] = f128::new(5.0);
    //      wilson_matrix[(1, 2)] = f128::new(8.0);
    //      wilson_matrix[(2, 1)] = f128::new(8.0);
    //      wilson_matrix[(1, 3)] = f128::new(7.0);
    //      wilson_matrix[(3, 1)] = f128::new(7.0);
    //      wilson_matrix[(2, 3)] = f128::new(9.0);
    //      wilson_matrix[(3, 2)] = f128::new(9.0);

    //      let decomposition_result = wilson_matrix.decompose_for_tropical(&settings).unwrap();
    //      let identity = &wilson_matrix * &decomposition_result.inverse;

    //      for row in 0..4 {
    //          for col in 0..4 {
    //              assert_approx_eq(
    //                  Into::<f64>::into(identity[(row, col)]),
    //                  if row == col { 1.0 } else { 0.0 },
    //                  EPSILON,
    //              );
    //          }
    //      }

    //      assert_approx_eq(
    //          Into::<f64>::into(decomposition_result.determinant),
    //          1.0,
    //          EPSILON,
    //      );
    //  }
}
