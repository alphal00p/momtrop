use crate::float::FloatLike;
use f128::f128;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Vector<T: FloatLike, const D: usize> {
    elements: [T; D],
}

impl<T: FloatLike, const D: usize> Vector<T, D> {
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
    pub fn from_slice(elements: &[T; D]) -> Self {
        Self {
            elements: *elements,
        }
    }

    #[inline]
    pub fn new() -> Self {
        Self {
            elements: [T::zero(); D],
        }
    }

    #[inline]
    pub fn squared(&self) -> T {
        let mut res = T::zero();
        for elem in self.elements {
            res += elem * elem
        }

        res
    }

    #[inline]
    pub fn dot(&self, rhs: &Self) -> T {
        let mut res = T::zero();

        for (&lhs_elem, &rhs_elem) in self.elements.iter().zip(rhs.elements.iter()) {
            res += lhs_elem * rhs_elem
        }

        res
    }

    // We are not going to do zero-dimensional qft!
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    #[inline]
    pub fn get_elements(&self) -> [T; D] {
        self.elements
    }
}

impl<T: FloatLike, const D: usize> Index<usize> for Vector<T, D> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl<T: FloatLike, const D: usize> IndexMut<usize> for Vector<T, D> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.elements[index]
    }
}

impl<const D: usize> Vector<f64, D> {
    pub fn upcast(&self) -> Vector<f128, D> {
        let mut elements = [f128::new(0.0); D];
        for (new_element, old_element) in elements.iter_mut().zip(self.elements.iter()) {
            *new_element = f128::new(*old_element);
        }

        Vector { elements }
    }
}

impl<T: FloatLike, const D: usize> Mul<T> for &Vector<T, D> {
    type Output = Vector<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut res = *self;

        for i in 0..D {
            res[i] *= rhs;
        }

        res
    }
}

impl<T: FloatLike, const D: usize> Add<&Vector<T, D>> for &Vector<T, D> {
    type Output = Vector<T, D>;

    #[inline]
    fn add(self, rhs: &Vector<T, D>) -> Self::Output {
        let mut res = Self::Output::new();

        for i in 0..D {
            res[i] += self[i] + rhs[i];
        }

        res
    }
}

impl<T: FloatLike, const D: usize> Sub<&Vector<T, D>> for &Vector<T, D> {
    type Output = Vector<T, D>;

    #[inline]
    fn sub(self, rhs: &Vector<T, D>) -> Self::Output {
        let mut res = Self::Output::new();

        for i in 0..D {
            res[i] += self[i] - rhs[i];
        }

        res
    }
}

impl<T: FloatLike, const D: usize> AddAssign for Vector<T, D> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self[i] += rhs[i];
        }
    }
}

impl<const D: usize> Vector<f128, D> {
    pub fn downcast(&self) -> Vector<f64, D> {
        let mut new_elements: [f64; D] = [0.0; D];
        for (new_element, out_element) in new_elements.iter_mut().zip(self.elements) {
            *new_element = out_element.into();
        }

        Vector {
            elements: new_elements,
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
        let vector: Vector<f64, 3> = Vector::new();

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
    fn test_upcast() {
        let vector = Vector::from_array([2.0, 3.0]);
        let vector_f128 = vector.upcast();

        assert_eq!(vector_f128.elements[0], f128::f128::new(2.0));
        assert_eq!(vector_f128.elements[1], f128::f128::new(3.0));
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
