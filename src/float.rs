use core::f64;
use ref_ops::{RefAdd, RefDiv, RefMul, RefNeg, RefSub};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[allow(clippy::wrong_self_convention)]
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
    fn one(&self) -> Self;
    fn ln(&self) -> Self;
    fn exp(&self) -> Self;
    fn cos(&self) -> Self;
    fn sin(&self) -> Self;
    fn powf(&self, power: &Self) -> Self;
    fn sqrt(&self) -> Self;
    fn from_isize(&self, value: isize) -> Self;
    fn from_f64(&self, value: f64) -> Self;
    fn inv(&self) -> Self;
    fn to_f64(&self) -> f64;
    fn zero(&self) -> Self;
    fn abs(&self) -> Self;
    #[allow(non_snake_case)]
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

//#[derive(Clone, Copy)]
//pub struct F<T: MomTropFloat>(pub T);
//
//impl<T: MomTropFloat> Add<&F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn add(self, rhs: &F<T>) -> Self::Output {
//        F(self.0.ref_add(&rhs.0))
//    }
//}
//
//impl<T: MomTropFloat> Sub<&F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn sub(self, rhs: &F<T>) -> Self::Output {
//        F(self.0.ref_sub(&rhs.0))
//    }
//}
//
//impl<T: MomTropFloat> Mul<&F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn mul(self, rhs: &F<T>) -> Self::Output {
//        F(self.0.ref_mul(&rhs.0))
//    }
//}
//
//impl<T: MomTropFloat> Div<&F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn div(self, rhs: &F<T>) -> Self::Output {
//        println!("calling ref_div");
//        F(self.0.ref_div(&rhs.0))
//    }
//}
//
//// additional addition traits
//impl<T: MomTropFloat> Add<F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn add(self, rhs: F<T>) -> Self::Output {
//        self + &rhs
//    }
//}
//
//impl<T: MomTropFloat> Add<&F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn add(self, rhs: &F<T>) -> Self::Output {
//        &self + rhs
//    }
//}
//
//impl<T: MomTropFloat> Add<F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn add(self, rhs: F<T>) -> Self::Output {
//        &self + &rhs
//    }
//}
//
//// additional subtraction traits
//impl<T: MomTropFloat> Sub<F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn sub(self, rhs: F<T>) -> Self::Output {
//        println!("recursing");
//        self - &rhs
//    }
//}
//
//impl<T: MomTropFloat> Sub<&F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn sub(self, rhs: &F<T>) -> Self::Output {
//        println!("recursing");
//        &self - rhs
//    }
//}
//
//impl<T: MomTropFloat> Sub<F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn sub(self, rhs: F<T>) -> Self::Output {
//        println!("recursing");
//        &self - &rhs
//    }
//}
//
//// additional multiplication traits
//impl<T: MomTropFloat> Mul<F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn mul(self, rhs: F<T>) -> Self::Output {
//        println!("recursing");
//        self * &rhs
//    }
//}
//
//impl<T: MomTropFloat> Mul<&F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn mul(self, rhs: &F<T>) -> Self::Output {
//        println!("recursing");
//        &self * rhs
//    }
//}
//
//impl<T: MomTropFloat> Mul<F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn mul(self, rhs: F<T>) -> Self::Output {
//        println!("recursing");
//        &self * &rhs
//    }
//}
//
//impl<T: MomTropFloat> Div<F<T>> for &F<T> {
//    type Output = F<T>;
//
//    fn div(self, rhs: F<T>) -> Self::Output {
//        println!("no_ref ref");
//        self / &rhs
//    }
//}
//
//impl<T: MomTropFloat> Div<&F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn div(self, rhs: &F<T>) -> Self::Output {
//        println!("ref no_ref");
//        &self / rhs
//    }
//}
//
//impl<T: MomTropFloat> Div<F<T>> for F<T> {
//    type Output = F<T>;
//
//    fn div(self, rhs: F<T>) -> Self::Output {
//        println!("no_ref no_ref");
//        &self / &rhs
//    }
//}
//
//impl<T: MomTropFloat> Neg for &F<T> {
//    type Output = F<T>;
//
//    fn neg(self) -> Self::Output {
//        self.ref_neg()
//    }
//}
//
//impl<T: MomTropFloat> Neg for F<T> {
//    type Output = F<T>;
//
//    fn neg(self) -> Self::Output {
//        -&self
//    }
//}
//
//impl<T: MomTropFloat> AddAssign<&F<T>> for F<T> {
//    fn add_assign(&mut self, rhs: &F<T>) {
//        *self = &*self + rhs;
//    }
//}
//
//impl<T: MomTropFloat> AddAssign<F<T>> for F<T> {
//    fn add_assign(&mut self, rhs: F<T>) {
//        *self = &*self + rhs;
//    }
//}
//
//impl<T: MomTropFloat> SubAssign<&F<T>> for F<T> {
//    fn sub_assign(&mut self, rhs: &F<T>) {
//        println!("recursing");
//        *self = &*self - rhs;
//    }
//}
//
//impl<T: MomTropFloat> SubAssign<F<T>> for F<T> {
//    fn sub_assign(&mut self, rhs: F<T>) {
//        println!("recursing");
//        *self = &*self - rhs;
//    }
//}
//
//impl<T: MomTropFloat> MulAssign<&F<T>> for F<T> {
//    fn mul_assign(&mut self, rhs: &F<T>) {
//        *self = &*self * rhs
//    }
//}
//
//impl<T: MomTropFloat> MulAssign<F<T>> for F<T> {
//    fn mul_assign(&mut self, rhs: F<T>) {
//        *self = &*self * rhs
//    }
//}
//
//impl<T: MomTropFloat> PartialEq for F<T> {
//    fn eq(&self, other: &Self) -> bool {
//        self.0 == other.0
//    }
//}
//
//impl<T: MomTropFloat> PartialOrd for F<T> {
//    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//        self.0.partial_cmp(&other.0)
//    }
//}
//
//impl<T: MomTropFloat> Debug for F<T> {
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        self.0.fmt(f)
//    }
//}
//
//impl<T: MomTropFloat> F<T> {
//    pub fn zero(&self) -> Self {
//        Self(self.0.from_isize(0))
//    }
//
//    pub fn one(&self) -> Self {
//        Self(self.0.from_isize(1))
//    }
//
//    pub fn from_isize(&self, value: isize) -> Self {
//        Self(self.0.from_isize(value))
//    }
//
//    pub fn from_f64(&self, value: f64) -> Self {
//        Self(self.0.from_f64(value))
//    }
//
//    pub fn abs(&self) -> Self {
//        if self < &self.zero() {
//            -self
//        } else {
//            self.clone()
//        }
//    }
//
//    pub fn sqrt(&self) -> Self {
//        Self(self.0.sqrt())
//    }
//
//    pub fn ln(&self) -> Self {
//        Self(self.0.ln())
//    }
//
//    pub fn inv(&self) -> Self {
//        Self(self.0.inv())
//    }
//
//    pub fn to_f64(&self) -> f64 {
//        self.0.to_f64()
//    }
//
//    #[allow(non_snake_case)]
//    pub fn PI(&self) -> Self {
//        Self(self.0.PI())
//    }
//
//    pub fn cos(&self) -> Self {
//        Self(self.0.cos())
//    }
//
//    pub fn sin(&self) -> Self {
//        Self(self.0.sin())
//    }
//
//    pub fn powf(&self, power: &Self) -> Self {
//        Self(self.0.powf(&power.0))
//    }
//}
//
//impl<T: MomTropFloat> From<T> for F<T> {
//    fn from(value: T) -> Self {
//        Self(value)
//    }
//}
