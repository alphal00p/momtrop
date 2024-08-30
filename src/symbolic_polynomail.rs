use itertools::Itertools;
use num::traits::sign;
use symbolica::{atom::Atom, domains::atom::AtomField, tensors::matrix::Matrix};

pub fn l_matrix_from_signature(signature_matrix: &[Vec<i64>]) -> Matrix<AtomField> {
    let loop_number = signature_matrix[0].len();
    let num_edges = signature_matrix.len();
    let symbolic_x_vec = (0..num_edges)
        .map(|e| Atom::parse(&format!("x{}", e)).unwrap())
        .collect_vec();

    let mut res = vec![vec![Atom::new(); loop_number]; loop_number];

    for i in 0..loop_number {
        for j in 0..loop_number {
            for e in 0..num_edges {
                res[i][j] = &res[i][j]
                    + &symbolic_x_vec[e]
                        * Atom::new_num(signature_matrix[e][i])
                        * Atom::new_num(signature_matrix[e][j]);
            }
        }
    }

    Matrix::from_nested_vec(res, AtomField {}).expect("Unable to construct matrix")
}

pub fn u_polynomial_from_signature(signature_matrix: &[Vec<i64>]) -> Atom {
    let l_matrix = l_matrix_from_signature(signature_matrix);
    l_matrix
        .det()
        .expect("Failed to take symbolic_determinant of l_matrix")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u_polynomial() {
        let triangle_sig = vec![vec![1]; 3];
        let target = Atom::parse("x0 + x1 + x2").unwrap();
        let u = u_polynomial_from_signature(&triangle_sig);
        assert_eq!(u, target);

        let double_triangle_sig = vec![vec![1, 0], vec![1, 0], vec![1, 1], vec![0, 1], vec![0, 1]];
        let target =
            Atom::parse("x0*x2 + x0*x3 + x0*x4 + x1*x2 + x1*x3 + x1*x4 + x2*x3 + x2*x4").unwrap();

        let u = u_polynomial_from_signature(&double_triangle_sig);
        let u = u.expand();
        assert_eq!(u, target);
    }
}
