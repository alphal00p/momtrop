use std::fmt::{Debug, Display, LowerExp};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, MulAssign, Sub, SubAssign};

use num::traits::{Float, FloatConst};
use num::traits::{Inv, One, Zero};

pub trait FloatLike:
    From<f64>
    + AddAssign
    + Div<Self, Output = Self>
    + Zero
    + One
    + Inv<Output = Self>
    + PartialOrd
    + PartialEq
    + Display
    + Sub<Self, Output = Self>
    + Clone
    + SubAssign
    + Float
    + MulAssign
    + Add<Self, Output = Self>
    + FloatConst
    + Into<f64>
    + Sum
    + Debug
    + LowerExp
{
}

impl FloatLike for f64 {}
impl FloatLike for f128::f128 {}
