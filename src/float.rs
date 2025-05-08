use core::f64;
use ref_ops::{RefAdd, RefDiv, RefMul, RefNeg, RefSub};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[allow(clippy::wrong_self_convention)]
/// Trait that allows momtrop to use arbitrary floating-point types supplied by the user.
/// Functions that return constants like `one()` require `self` has argument to facilitate more
/// advanced arbitrary precision types.
pub trait MomTropFloat:
    for<'a> RefAdd<&'a Self, Output = Self>
    + for<'a> RefSub<&'a Self, Output = Self>
    + for<'a> RefMul<&'a Self, Output = Self>
    + for<'a> RefDiv<&'a Self, Output = Self>
    + for<'a> RefAdd<Self, Output = Self>
    + for<'a> RefSub<Self, Output = Self>
    + for<'a> RefMul<Self, Output = Self>
    + for<'a> RefDiv<Self, Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> RefNeg<Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Neg<Output = Self>
    + Clone
    + PartialEq
    + Debug
    + PartialOrd
{
    /// Returns 1.
    fn one(&self) -> Self;
    /// Return natural logarithm of `self`.
    fn ln(&self) -> Self;
    /// Return exponential of `self`.
    fn exp(&self) -> Self;
    /// Return  cosine of `self`.
    fn cos(&self) -> Self;
    /// Return  sine of `self`.
    fn sin(&self) -> Self;
    /// Raise `self` to `power`.
    fn powf(&self, power: &Self) -> Self;
    /// Return `sqrt` of `self`.
    fn sqrt(&self) -> Self;
    /// Create a new value from an `isize`.
    fn from_isize(&self, value: isize) -> Self;
    /// Create a new value from a `f64``.
    fn from_f64(&self, value: f64) -> Self;
    /// Return inverse of `self`.
    fn inv(&self) -> Self;
    /// Convert `self` to `f64`.
    fn to_f64(&self) -> f64;
    /// Returns 0.
    fn zero(&self) -> Self;
    /// Return absolute value of `self`.
    fn abs(&self) -> Self;
    #[allow(non_snake_case)]
    /// Return pi.
    fn PI(&self) -> Self;
}

impl MomTropFloat for f64 {
    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    fn cos(&self) -> Self {
        f64::cos(*self)
    }

    fn sin(&self) -> Self {
        f64::sin(*self)
    }

    fn powf(&self, power: &Self) -> Self {
        f64::powf(*self, *power)
    }

    fn from_f64(&self, value: f64) -> Self {
        value
    }

    fn from_isize(&self, value: isize) -> Self {
        value as f64
    }

    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    fn inv(&self) -> Self {
        1.0 / self
    }

    fn to_f64(&self) -> f64 {
        *self
    }

    fn PI(&self) -> Self {
        f64::consts::PI
    }

    fn zero(&self) -> Self {
        0.0
    }

    fn one(&self) -> Self {
        1.0
    }

    fn abs(&self) -> Self {
        f64::abs(*self)
    }
}
