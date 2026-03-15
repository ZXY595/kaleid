use extension_traits::extension;
use nalgebra::{
    Cholesky, DefaultAllocator, DimName, Matrix, OMatrix, Storage, Vector, allocator::Allocator,
};

use crate::{DoF, Element, error_state::ErrorState};

impl<D: ErrorState> crate::Covariance<D>
where
    DefaultAllocator: Allocator<D::DoF, D::DoF> + Allocator<D::DoF>,
{
    pub fn observe<DD: DoF>(
        &mut self,
        states: &mut D,
        measurement: Vector<Element, DD::DoF, impl Storage<Element, DD::DoF>>,
        noise: Vector<Element, DD::DoF, impl Storage<Element, DD::DoF>>,
        cross_cov: Matrix<Element, D::DoF, DD::DoF, impl Storage<Element, D::DoF, DD::DoF>>,
        cross_cov_tr: Matrix<Element, DD::DoF, D::DoF, impl Storage<Element, DD::DoF, D::DoF>>,
        innovation_cov: Matrix<Element, DD::DoF, DD::DoF, impl Storage<Element, DD::DoF, DD::DoF>>,
    ) where
        DefaultAllocator: Allocator<DD::DoF, DD::DoF>
            + Allocator<DD::DoF>
            + Allocator<D::DoF, DD::DoF>
            + Allocator<DD::DoF, D::DoF>,
    {
        let kalman_gain = cross_cov
            * innovation_cov
                .diagonal_add(noise)
                .cholesky_inverse_with_substitute();

        states.inject(&kalman_gain * measurement);
        self.0 -= kalman_gain * cross_cov_tr;
    }
}

#[extension(trait DiagonalAdd)]
impl<D: DimName, S> Matrix<Element, D, D, S>
where
    S: Storage<Element, D, D>,
    DefaultAllocator: Allocator<D, D>,
{
    #[inline]
    fn diagonal_add(
        self,
        diagnoal: Vector<Element, D, impl Storage<Element, D>>,
    ) -> OMatrix<Element, D, D> {
        let mut result = self.into_owned();
        (0..D::DIM).for_each(|i| result[(i, i)] += diagnoal[i]);
        result
    }
}

#[extension(trait CholeskyInverse)]
impl<D: DimName> OMatrix<Element, D, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    fn cholesky_inverse_with_substitute(self) -> OMatrix<Element, D, D> {
        let cholesky = Cholesky::new_with_substitute(self, 0.0001);
        // # SAFETY:
        //
        // this is safe because the value of `T::SUBSTITUTE` is positive definite
        // and the Cholesky decomposition is always successful
        let cholesky = unsafe { cholesky.unwrap_unchecked() };
        cholesky.inverse()
    }
}
