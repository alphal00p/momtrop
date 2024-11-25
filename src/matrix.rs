use smallvec::SmallVec;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

use crate::{float::FloatLike, TropicalSamplingSettings};
use f128::f128;

#[derive(Debug, Clone, Default)]
/// square symmetric matrix for use in the tropical sampling algorithm
pub struct SquareMatrix<T> {
    data: SmallVec<[T; 36]>, // this allows us to store the L matrix on the stack for 6 loops and below
    dim: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum MatrixError {
    ZeroDet,
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

impl<'a, T: FloatLike> Mul<&'a SquareMatrix<T>> for &'a SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = SquareMatrix::new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                for k in 0..self.dim {
                    result[(row, col)] += self[(row, k)] * rhs[(k, col)];
                }
            }
        }

        result
    }
}

impl<T: FloatLike> Mul<T> for SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = SquareMatrix::new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                result[(row, col)] = self[(row, col)] * rhs;
            }
        }

        result
    }
}

impl<'a, T: FloatLike> Add<&'a SquareMatrix<T>> for &'a SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = SquareMatrix::new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                result[(row, col)] = self[(row, col)] + rhs[(row, col)];
            }
        }

        result
    }
}

impl<'a, T: FloatLike> Sub<&'a SquareMatrix<T>> for &'a SquareMatrix<T> {
    type Output = SquareMatrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = SquareMatrix::new_zeros(self.dim);

        for row in 0..self.dim {
            for col in 0..self.dim {
                result[(row, col)] = self[(row, col)] - rhs[(row, col)];
            }
        }

        result
    }
}

impl<T: FloatLike> SquareMatrix<T> {
    #[must_use]
    pub fn new_zeros(dim: usize) -> Self {
        Self {
            data: SmallVec::from_elem(T::zero(), dim * dim),
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
            println!("starting cholesky");
        }

        // start cholesky decomposition
        let mut q: SquareMatrix<T> = SquareMatrix::new_zeros(self.dim);

        for i in 0..self.dim {
            let mut diagonal_entry_squared = self[(i, i)];
            for j in 0..i {
                diagonal_entry_squared -= q[(i, j)] * q[(i, j)];
            }

            let diagonal_entry = diagonal_entry_squared.sqrt();
            q[(i, i)] = diagonal_entry;

            for j in i + 1..self.dim {
                let mut entry = self[(i, j)];
                for k in 0..i {
                    entry -= q[(i, k)] * q[(j, k)];
                }
                q[(j, i)] = entry / diagonal_entry;
            }
        }
        // end cholesky decomposition

        // compute the determinant of Q and store the invesrses of the diagonal elements
        let mut det_q = T::one();
        let mut inverse_diagonal_entries = SmallVec::<[T; 6]>::new();

        for i in 0..self.dim {
            let q_ii = q[(i, i)];
            det_q *= q_ii;
            inverse_diagonal_entries.push(q_ii.inv());
        }

        (0..self.dim).fold(T::one(), |acc, i| acc * q[(i, i)]);
        let determinant = det_q * det_q;

        if det_q == T::zero() {
            return Err(MatrixError::ZeroDet);
        }

        // the matrix N is defined through Q = D(I + N)
        let mut n_matrix: SquareMatrix<T> = SquareMatrix::new_zeros(self.dim);
        for row in 1..self.dim {
            let inverse_diagonal_element = inverse_diagonal_entries[row];
            for col in 0..row {
                n_matrix[(row, col)] = inverse_diagonal_element * q[(row, col)];
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

        let n_sum = powers_of_n.iter().enumerate().fold(
            SquareMatrix::new_zeros(self.dim),
            |acc, (i, mat)| {
                if i % 2 == 0 {
                    &acc - mat
                } else {
                    &acc + mat
                }
            },
        );

        let mut inverse_q = n_sum;
        for row in 0..self.dim {
            inverse_q[(row, row)] += T::one();
            for col in 0..self.dim {
                inverse_q[(row, col)] *= inverse_diagonal_entries[col];
            }
        }

        let mut q_transposed_inverse = SquareMatrix::new_zeros(self.dim);
        let mut q_transposed = SquareMatrix::new_zeros(self.dim);
        for row in 0..self.dim {
            for col in 0..self.dim {
                q_transposed_inverse[(row, col)] = inverse_q[(col, row)];
                q_transposed[(row, col)] = q[(col, row)];
            }
        }

        let inverse = &q_transposed_inverse * &inverse_q;

        if let Some(tolerance) = settings.matrix_stability_test {
            if settings.print_debug_info {
                println!("Performing matrix inversion quality");
            }

            let approx_idendity = &inverse * self;
            let true_identity = Self::new_identity(self.dim);
            let zero = &approx_idendity - &true_identity;
            let error = zero.l21_norm();

            if settings.print_debug_info {
                println!("error: {:+16e}", error);
            }

            if Into::<f64>::into(error) > tolerance {
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

    fn new_identity(dim: usize) -> Self {
        let mut res = Self::new_zeros(dim);
        for i in 0..dim {
            res[(i, i)] = T::one();
        }
        res
    }

    fn l21_norm(&self) -> T {
        let mut res = T::zero();

        for j in 0..self.dim {
            let mut vec_norm = T::zero();
            for i in 0..self.dim {
                vec_norm += self[(i, j)] * self[(i, j)];
            }
            res += vec_norm.sqrt();
        }

        res
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

impl SquareMatrix<f128> {
    pub fn downcast(&self) -> SquareMatrix<f64> {
        SquareMatrix {
            data: self.data.iter().map(|&x| x.into()).collect(),
            dim: self.dim,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecompositionResult<T> {
    pub determinant: T,
    pub inverse: SquareMatrix<T>,
    pub q_transposed: SquareMatrix<T>,
    pub q_transposed_inverse: SquareMatrix<T>,
}

impl DecompositionResult<f128> {
    pub fn downcast(&self) -> DecompositionResult<f64> {
        DecompositionResult {
            determinant: self.determinant.into(),
            q_transposed: self.q_transposed.downcast(),
            inverse: self.inverse.downcast(),
            q_transposed_inverse: self.q_transposed_inverse.downcast(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_decompose_for_tropical_2x2() {
        let mut test_matrix = SquareMatrix::new_zeros(2);
        let settings = TropicalSamplingSettings::default();

        test_matrix[(0, 0)] = 2.0;
        test_matrix[(1, 1)] = 4.0;
        test_matrix[(0, 1)] = 1.0;
        test_matrix[(1, 0)] = 1.0;

        let decomposition_result = test_matrix.decompose_for_tropical(&settings).unwrap();

        assert_approx_eq(decomposition_result.determinant, 7.0, EPSILON);

        let identity = &test_matrix * &decomposition_result.inverse;

        assert_approx_eq(identity[(0, 0)], 1.0, EPSILON);
        assert_approx_eq(identity[(0, 1)], 0.0, EPSILON);
        assert_approx_eq(identity[(1, 0)], 0.0, EPSILON);
        assert_approx_eq(identity[(1, 1)], 1.0, EPSILON);
    }

    #[test]
    fn test_decompose_for_tropical_f128_2x2() {
        let mut test_matrix = SquareMatrix::new_zeros(2);
        let settings = TropicalSamplingSettings::default();

        test_matrix[(0, 0)] = f128::new(2.0);
        test_matrix[(1, 1)] = f128::new(4.0);
        test_matrix[(0, 1)] = f128::new(1.0);
        test_matrix[(1, 0)] = f128::new(1.0);

        let decomposition_result = test_matrix.decompose_for_tropical(&settings).unwrap();

        assert_approx_eq(
            Into::<f64>::into(decomposition_result.determinant),
            7.0,
            EPSILON,
        );

        let identity = &test_matrix * &decomposition_result.inverse;

        assert_approx_eq(Into::<f64>::into(identity[(0, 0)]), 1.0, EPSILON);
        assert_approx_eq(Into::<f64>::into(identity[(0, 1)]), 0.0, EPSILON);
        assert_approx_eq(Into::<f64>::into(identity[(1, 0)]), 0.0, EPSILON);
        assert_approx_eq(Into::<f64>::into(identity[(1, 1)]), 1.0, EPSILON);
    }

    #[test]
    fn test_decompose_for_tropical_4x4() {
        let mut wilson_matrix = SquareMatrix::new_zeros(4);
        let settings = TropicalSamplingSettings::default();

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
                    identity[(row, col)],
                    if row == col { 1.0 } else { 0.0 },
                    EPSILON,
                );
            }
        }

        assert_approx_eq(decomposition_result.determinant, 1.0, EPSILON);
    }

    #[test]
    fn test_decompose_for_tropical_4x4_f128() {
        let mut wilson_matrix = SquareMatrix::new_zeros(4);
        let settings = TropicalSamplingSettings::default();

        wilson_matrix[(0, 0)] = f128::new(5.0);
        wilson_matrix[(1, 1)] = f128::new(10.);
        wilson_matrix[(2, 2)] = f128::new(10.);
        wilson_matrix[(3, 3)] = f128::new(10.);
        wilson_matrix[(0, 1)] = f128::new(7.0);
        wilson_matrix[(1, 0)] = f128::new(7.0);
        wilson_matrix[(0, 2)] = f128::new(6.0);
        wilson_matrix[(2, 0)] = f128::new(6.0);
        wilson_matrix[(0, 3)] = f128::new(5.0);
        wilson_matrix[(3, 0)] = f128::new(5.0);
        wilson_matrix[(1, 2)] = f128::new(8.0);
        wilson_matrix[(2, 1)] = f128::new(8.0);
        wilson_matrix[(1, 3)] = f128::new(7.0);
        wilson_matrix[(3, 1)] = f128::new(7.0);
        wilson_matrix[(2, 3)] = f128::new(9.0);
        wilson_matrix[(3, 2)] = f128::new(9.0);

        let decomposition_result = wilson_matrix.decompose_for_tropical(&settings).unwrap();
        let identity = &wilson_matrix * &decomposition_result.inverse;

        for row in 0..4 {
            for col in 0..4 {
                assert_approx_eq(
                    Into::<f64>::into(identity[(row, col)]),
                    if row == col { 1.0 } else { 0.0 },
                    EPSILON,
                );
            }
        }

        assert_approx_eq(
            Into::<f64>::into(decomposition_result.determinant),
            1.0,
            EPSILON,
        );
    }
}
