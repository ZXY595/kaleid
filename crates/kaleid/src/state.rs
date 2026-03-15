use nalgebra::{Const, DimName, Scalar};

use crate::ImplVector;

pub mod bundle;
pub mod common;

pub trait KFState {
    type Element: Scalar;
    type Dim: DimName + IntoArray;

    fn gain(&mut self, kalman_gain: ImplVector!(Self::Element, Self::Dim)) {
        let _ = kalman_gain;
    }
}

pub type StateDim<S> = <S as KFState>::Dim;

pub const fn state_dim<S: KFState>() -> usize {
    StateDim::<S>::DIM
}

pub(crate) trait IntoArray {
    type Array<T>;
}

impl<const N: usize> IntoArray for Const<N> {
    type Array<T> = [T; N];
}

pub(crate) type ConstArray<T, N> = <N as IntoArray>::Array<T>;
