use f128::f128;
use itertools::Itertools;

use crate::float::FloatLike;
use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(Clone, Debug)]
pub struct Vector<T: FloatLike> {
    elements: Vec<T>,
}

impl<T: FloatLike> Vector<T> {
    pub fn from_vec(elements: Vec<T>) -> Self {
        Self { elements }
    }

    pub fn from_slice(elements: &[T]) -> Self {
        Self {
            elements: elements.iter().copied().collect_vec(),
        }
    }

    pub fn new(dimension: usize) -> Self {
        Self {
            elements: vec![T::zero(); dimension],
        }
    }

    pub fn squared(&self) -> T {
        self.elements.iter().map(|&x| x * x).sum::<T>()
    }

    pub fn dot(&self, rhs: &Self) -> T {
        self.elements
            .iter()
            .zip(rhs.elements.iter())
            .map(|(&left, &right)| left * right)
            .sum::<T>()
    }
}

impl Vector<f64> {
    pub fn upcast(&self) -> Vector<f128> {
        Vector {
            elements: self.elements.iter().copied().map(f128::new).collect_vec(),
        }
    }
}

impl<T: FloatLike> Mul<T> for &Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            elements: self.elements.iter().map(|&x| rhs * x).collect_vec(),
        }
    }
}

impl<T: FloatLike> Add<&Vector<T>> for &Vector<T> {
    type Output = Vector<T>;

    fn add(self, rhs: &Vector<T>) -> Self::Output {
        Self::Output {
            elements: self
                .elements
                .iter()
                .zip(rhs.elements.iter())
                .map(|(&left, &right)| left + right)
                .collect_vec(),
        }
    }
}

impl<T: FloatLike> Sub<&Vector<T>> for &Vector<T> {
    type Output = Vector<T>;

    fn sub(self, rhs: &Vector<T>) -> Self::Output {
        Self::Output {
            elements: self
                .elements
                .iter()
                .zip(rhs.elements.iter())
                .map(|(&left, &right)| left - right)
                .collect_vec(),
        }
    }
}

impl<T: FloatLike> AddAssign for Vector<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.elements
            .iter_mut()
            .zip(rhs.elements.iter())
            .for_each(|(element, &rhs_element)| *element += rhs_element);
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::Vector;

    #[test]
    fn test_from_vec() {
        let test_vec = vec![1., 2., 3.];
        let vector = Vector::from_vec(test_vec);

        assert_eq!(vector.elements[0], 1.);
        assert_eq!(vector.elements[1], 2.);
        assert_eq!(vector.elements[2], 3.);
    }

    #[test]
    fn test_from_slice() {
        let test_vec = vec![1., 2., 3.];
        let vector = Vector::from_slice(&test_vec);

        assert_eq!(vector.elements[0], 1.);
        assert_eq!(vector.elements[1], 2.);
        assert_eq!(vector.elements[2], 3.);
    }

    #[test]
    fn test_new() {
        let vector: Vector<f64> = Vector::new(3);

        assert_eq!(vector.elements[0], 0.0);
        assert_eq!(vector.elements[0], 0.0);
        assert_eq!(vector.elements[0], 0.0);
    }

    #[test]
    fn test_squared() {
        let pythagoras = Vector::from_vec(vec![3.0, 4.0]);
        let squared = pythagoras.squared();
        assert_eq!(squared, 25.);
    }

    #[test]
    fn test_dot() {
        let vector1 = Vector::from_vec(vec![1.0, 2.0]);
        let vector2 = Vector::from_vec(vec![-1.0, 0.5]);
        let dot = vector1.dot(&vector2);
        assert_eq!(dot, 0.0);
    }

    #[test]
    fn test_upcast() {
        let vector = Vector::from_vec(vec![2.0, 3.0]);
        let vector_f128 = vector.upcast();

        assert_eq!(vector_f128.elements[0], f128::f128::new(2.0));
        assert_eq!(vector_f128.elements[1], f128::f128::new(3.0));
    }

    #[test]
    fn test_mul() {
        let vector = Vector::from_vec(vec![2.0, 4.0]);
        let vector_2 = &vector * 4.0;
        assert_eq!(vector_2.elements[0], 8.0);
        assert_eq!(vector_2.elements[1], 16.0);
    }

    #[test]
    fn test_add() {
        let vector1 = Vector::from_vec(vec![1.0, 2.0]);
        let vector2 = Vector::from_vec(vec![1.0, -2.0]);

        let vector3 = &vector1 + &vector2;

        assert_eq!(vector3.elements[0], 2.0);
        assert_eq!(vector3.elements[1], 0.0);
    }

    #[test]
    fn test_add_assign() {
        let mut vector1 = Vector::from_vec(vec![1.0, 2.0]);
        let vector2 = Vector::from_vec(vec![1.0, -2.0]);
        vector1 += vector2;

        assert_eq!(vector1.elements[0], 2.0);
        assert_eq!(vector1.elements[1], 0.0);
    }
}
