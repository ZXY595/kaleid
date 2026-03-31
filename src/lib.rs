#![forbid(clippy::undocumented_unsafe_blocks)]

pub mod builder;
pub mod error_state;
pub mod observe;
pub mod predict;

use derive_deref::{Deref, DerefMut};
use nalgebra::{DefaultAllocator, OMatrix, allocator::Allocator};
pub type Element = f64;
pub use nalgebra::Const;

use crate::error_state::ErrorState;

#[derive(Deref, DerefMut)]
pub struct Covariance<D: ErrorState>(OMatrix<Element, D::DoF, D::DoF>)
where
    DefaultAllocator: Allocator<D::DoF, D::DoF>;

impl<D: ErrorState> Covariance<D>
where
    DefaultAllocator: Allocator<D::DoF, D::DoF>,
{
    pub fn new() -> Self {
        Self(OMatrix::<Element, D::DoF, D::DoF>::zeros())
    }
}

impl<D: ErrorState> Default for Covariance<D>
where
    DefaultAllocator: Allocator<D::DoF, D::DoF>,
{
    fn default() -> Self {
        Self::new()
    }
}
