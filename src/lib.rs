pub mod error_state;
pub mod extract;
pub mod observe;
pub mod predict;

use derive_deref::{Deref, DerefMut};
use nalgebra::{DefaultAllocator, DimName, DimNameAdd, DimNameSum, OMatrix, allocator::Allocator};
pub type Element = f64;
pub use nalgebra::Const;

type DoFMatrix<D> = OMatrix<Element, <D as DoF>::DoF, <D as DoF>::DoF>;

pub trait DoF {
    type DoF: DimName;
}

#[derive(Deref, DerefMut)]
pub struct Covariance<D: DoF>(DoFMatrix<D>)
where
    DefaultAllocator: Allocator<D::DoF, D::DoF>;

impl<D: DoF> Covariance<D>
where
    DefaultAllocator: Allocator<D::DoF, D::DoF>,
{
    pub fn new() -> Self {
        Self(DoFMatrix::<D>::zeros())
    }
}

impl<D: DoF> Default for Covariance<D>
where
    DefaultAllocator: Allocator<D::DoF, D::DoF>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl DoF for () {
    type DoF = Const<0>;
}

impl<T1: DoF, T2: DoF> DoF for (T1, T2)
where
    T1::DoF: DimNameAdd<T2::DoF>,
{
    type DoF = DimNameSum<T1::DoF, T2::DoF>;
}
