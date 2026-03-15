use std::ops::{Deref, DerefMut};

use nalgebra::{DefaultAllocator, OMatrix, Scalar, allocator::Allocator};
use num_traits::{One, Zero};
use simba::scalar::SupersetOf;

use super::Cov;
use crate::state::KFState;

#[derive(Debug, Clone)]
pub struct Uncertained<S: KFState>
where
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub state: S,
    /// The covariance matrix of the state.
    pub cov: Cov<S>,
}

impl<S: KFState> Deref for Uncertained<S>
where
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<S: KFState> DerefMut for Uncertained<S>
where
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<S> Uncertained<S>
where
    S: KFState<Element: One + Zero + SupersetOf<f64>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub fn new(state: S) -> Self {
        let cov = OMatrix::from_diagonal_element(nalgebra::convert(1e-6));
        Self::new_with_cov(state, cov)
    }

    pub const fn new_with_cov(state: S, cov: OMatrix<S::Element, S::Dim, S::Dim>) -> Self {
        Self { state, cov }
    }
}

impl<S> Default for Uncertained<S>
where
    S: KFState + Default,
    DefaultAllocator: Allocator<S::Dim, S::Dim, Buffer<S::Element>: Default>,
{
    fn default() -> Self {
        Self {
            state: Default::default(),
            cov: OMatrix::default(),
        }
    }
}

impl<S> Uncertained<&S>
where
    for<'s> &'s S: KFState<Element: Scalar>,
    DefaultAllocator: for<'s> Allocator<<&'s S as KFState>::Dim, <&'s S as KFState>::Dim>,
{
    pub fn as_deref_ref(&self) -> &S {
        self.state
    }
}
