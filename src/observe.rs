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
        cross_cov_tr: &Matrix<Element, DD::DoF, D::DoF, impl Storage<Element, DD::DoF, D::DoF>>,
        innovation_cov: Matrix<Element, DD::DoF, DD::DoF, impl Storage<Element, DD::DoF, DD::DoF>>,
    ) where
        DefaultAllocator: Allocator<DD::DoF, DD::DoF>
            + Allocator<DD::DoF>
            + Allocator<D::DoF, DD::DoF>
            + Allocator<DD::DoF, D::DoF>,
    {
        // S * K^T = HP
        let kalman_gain = innovation_cov
            .diagonal_add(noise)
            .cholesky_solve_with_substitute(cross_cov_tr)
            .transpose();

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
        let mut temp = self.into_owned();
        for i in 0..D::DIM {
            // # SAFETY:
            //
            // D::DIM implements `DimName` which means it is a type level constant,
            // temp[i, i] and diagnoal[i] is always safe to access in range.
            unsafe { *temp.get_unchecked_mut((i, i)) += diagnoal.vget_unchecked(i) }
        }
        temp
    }
}

#[extension(trait CholeskySolve)]
impl<D: DimName> OMatrix<Element, D, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    fn cholesky_solve_with_substitute<DD: DimName>(
        self,
        b: &Matrix<Element, D, DD, impl Storage<Element, D, DD>>,
    ) -> OMatrix<Element, D, DD>
    where
        DefaultAllocator: Allocator<D, DD>,
    {
        const SUBSTITUTE: Element = 0.0001;
        let cholesky = Cholesky::new_with_substitute(self, SUBSTITUTE);
        // # SAFETY:
        //
        // this is safe because the value of `SUBSTITUTE` is positive definite
        // and the Cholesky decomposition is always successful
        unsafe { cholesky.unwrap_unchecked() }.solve(b)
    }
}
