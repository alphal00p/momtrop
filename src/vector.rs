use crate::float::MomTropFloat;
use std::{
    array,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub},
};

#[derive(Clone, Copy, Debug)]
/// Vector struct, used to return loop momenta and receive the shifts of propagators.
pub struct Vector<T: MomTropFloat, const D: usize> {
    elements: [T; D],
}

impl<T: MomTropFloat, const D: usize> Vector<T, D> {
    #[inline]
    pub fn from_array(elements: [T; D]) -> Self {
        Self { elements }
    }

    #[inline]
    pub fn from_vec(elements: Vec<T>) -> Self {
        Self {
            elements: elements
                .try_into()
                .unwrap_or_else(|_| panic!("invalid dimension")),
        }
    }

    #[inline]
    /// Create a zero valued scalar from a vector
    pub fn zero(&self) -> T {
        self.elements[0].zero()
    }

    #[inline]
    pub fn from_slice(elements: &[T; D]) -> Self {
        Self {
            elements: elements.clone(),
        }
    }

    #[inline]
    /// Create a zero vector.
    pub fn new(&self) -> Self {
        Self {
            elements: self.elements.each_ref().map(|value| value.zero()),
        }
    }

    #[inline]
    /// Create a zero vector using `builder.zero()`.
    pub fn new_from_num(builder: &T) -> Self {
        Self {
            elements: array::from_fn(|_| builder.zero()),
        }
    }

    #[inline]
    /// Compute the square of a vector.
    pub fn squared(&self) -> T {
        self.elements
            .iter()
            .fold(self.elements[0].zero(), |acc, x| acc + x.ref_mul(x))
    }

    #[inline]
    /// Compute the dot product of two vectors.
    pub fn dot(&self, rhs: &Self) -> T {
        self.elements
            .iter()
            .zip(rhs.elements.iter())
            .fold(self.elements[0].zero(), |acc, (left, right)| {
                acc + left.ref_mul(right)
            })
    }

    // We are not going to do zero-dimensional qft!
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    #[inline]
    /// Get the array containing the elements of the vector.
    pub fn get_elements(&self) -> [T; D] {
        self.elements.clone()
    }
}

impl<T: MomTropFloat, const D: usize> Index<usize> for Vector<T, D> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T: MomTropFloat, const D: usize> IndexMut<usize> for Vector<T, D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<T: MomTropFloat, const D: usize> Mul<T> for &Vector<T, D> {
    type Output = Vector<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            elements: self.elements.each_ref().map(|elem| elem.ref_mul(&rhs)),
        }
    }
}

impl<T: MomTropFloat, const D: usize> Mul<&T> for &Vector<T, D> {
    type Output = Vector<T, D>;

    fn mul(self, rhs: &T) -> Self::Output {
        Self::Output {
            elements: self.elements.each_ref().map(|elem| elem.ref_mul(rhs)),
        }
    }
}

impl<T: MomTropFloat, const D: usize> Add<&Vector<T, D>> for &Vector<T, D> {
    type Output = Vector<T, D>;

    #[inline]
    fn add(self, rhs: &Vector<T, D>) -> Self::Output {
        //Self::Output {
        //    elements: self
        //        .elements
        //        .iter()
        //        .zip(&rhs.elements)
        //        .map(|(left, right)| left + right)
        //        .collect_vec()
        //        .try_into()
        //        .unwrap_or_else(|_| unreachable!()),
        //}

        Self::Output {
            elements: array::from_fn(|i| self[i].ref_add(&rhs[i])),
        }
    }
}

impl<T: MomTropFloat, const D: usize> Sub<&Vector<T, D>> for &Vector<T, D> {
    type Output = Vector<T, D>;

    #[inline]
    fn sub(self, rhs: &Vector<T, D>) -> Self::Output {
        //Self::Output {
        //    elements: self
        //        .elements
        //        .iter()
        //        .zip(&rhs.elements)
        //        .map(|(left, right)| left - right)
        //        .collect_vec()
        //        .try_into()
        //        .unwrap_or_else(|_| unreachable!()),
        //}

        Self::Output {
            elements: array::from_fn(|i| self[i].ref_sub(&rhs[i])),
        }
    }
}

impl<T: MomTropFloat, const D: usize> AddAssign for Vector<T, D> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self[i] += &rhs[i];
        }
    }
}

#[cfg(test)]
mod tests {

    use super::Vector;

    #[test]
    fn test_from_vec() {
        let test_vec = [1., 2., 3.];
        let vector = Vector::from_array(test_vec);

        assert_eq!(vector.elements[0], 1.);
        assert_eq!(vector.elements[1], 2.);
        assert_eq!(vector.elements[2], 3.);
    }

    #[test]
    fn test_from_slice() {
        let test_vec = [1., 2., 3.];
        let vector = Vector::from_slice(&test_vec);

        assert_eq!(vector.elements[0], 1.);
        assert_eq!(vector.elements[1], 2.);
        assert_eq!(vector.elements[2], 3.);
    }

    #[test]
    fn test_new() {
        let vector = Vector::from_array([1., 2., 3.]).new();

        assert_eq!(vector.elements[0], 0.0);
        assert_eq!(vector.elements[1], 0.0);
        assert_eq!(vector.elements[2], 0.0);
    }

    #[test]
    fn test_squared() {
        let pythagoras = Vector::from_array([3.0, 4.0]);
        let squared = pythagoras.squared();
        assert_eq!(squared, 25.);
    }

    #[test]
    fn test_dot() {
        let vector1 = Vector::from_array([1.0, 2.0]);
        let vector2 = Vector::from_array([-1.0, 0.5]);
        let dot = vector1.dot(&vector2);
        assert_eq!(dot, 0.0);
    }

    #[test]
    fn test_mul() {
        let vector = Vector::from_array([2.0, 4.0]);
        let vector_2 = &vector * 4.0;
        assert_eq!(vector_2.elements[0], 8.0);
        assert_eq!(vector_2.elements[1], 16.0);
    }

    #[test]
    fn test_add() {
        let vector1 = Vector::from_array([1.0, 2.0]);
        let vector2 = Vector::from_array([1.0, -2.0]);

        let vector3 = &vector1 + &vector2;

        assert_eq!(vector3.elements[0], 2.0);
        assert_eq!(vector3.elements[1], 0.0);
    }

    #[test]
    fn test_add_assign() {
        let mut vector1 = Vector::from_array([1.0, 2.0]);
        let vector2 = Vector::from_array([1.0, -2.0]);
        vector1 += vector2;

        assert_eq!(vector1.elements[0], 2.0);
        assert_eq!(vector1.elements[1], 0.0);
    }
}
