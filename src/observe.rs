use extension_traits::extension;
use nalgebra::{
    Cholesky, Const, DefaultAllocator, Dim, DimName, Matrix, MatrixView, OMatrix, Storage, Vector,
    ViewStorage, allocator::Allocator,
};

use crate::{Covariance, Element, error_state::ErrorState};

impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF> + Allocator<S::DoF>,
{
    pub fn observe<D: Dim>(
        &mut self,
        states: &mut S,
        measurement: &Vector<Element, D, impl Storage<Element, D>>,
        noise: &Vector<Element, D, impl Storage<Element, D>>,
        cross_cov_tr: &Matrix<Element, D, S::DoF, impl Storage<Element, D, S::DoF>>,
        innovation_cov: Matrix<Element, D, D, impl Storage<Element, D, D>>,
    ) where
        DefaultAllocator: Allocator<D, D> + Allocator<S::DoF, D> + Allocator<D, S::DoF>,
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
impl<D: Dim, S> Matrix<Element, D, D, S>
where
    S: Storage<Element, D, D>,
    DefaultAllocator: Allocator<D, D>,
{
    #[inline]
    fn diagonal_add(
        self,
        diagnoal: &Vector<Element, D, impl Storage<Element, D>>,
    ) -> OMatrix<Element, D, D> {
        let (nrows, ncols) = self.shape();
        let mut temp = self.into_owned();
        for i in 0..nrows.min(ncols) {
            // # SAFETY:
            //
            // temp[i, i] and diagnoal[i] is always safe to access in range.
            unsafe { *temp.get_unchecked_mut((i, i)) += diagnoal.vget_unchecked(i) }
        }
        temp
    }
}

#[extension(trait CholeskySolve)]
impl<D: Dim> OMatrix<Element, D, D>
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

impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF>,
{
    /// You must ensure that `D1` and `D2` is contiguous in `S`, otherwise you will get a panic
    pub fn block<D1: ErrorState, D2: ErrorState>(
        &self,
    ) -> MatrixView<'_, Element, D1::DoF, D2::DoF, Const<1>, S::DoF> {
        let start = (S::offset_of::<D1>().unwrap(), S::offset_of::<D2>().unwrap());

        // # Safety:
        //
        // `<S as DoF>::DoF` implements DimName, which means `RStride` always equal to `Const<1>`,
        // and `CStride` equal to `S::DoF`.
        unsafe {
            let data = ViewStorage::new_with_strides_unchecked(
                &self.data,
                start,
                (D1::DoF::name(), D2::DoF::name()),
                (Const::<1>, S::DoF::name()),
            );
            Matrix::from_data_statically_unchecked(data)
        }
    }

    /// You must ensure that `D` is contiguous in `S`, otherwise you will get a panic
    pub fn rows<D: ErrorState>(&self) -> MatrixView<'_, Element, D::DoF, S::DoF, Const<1>, S::DoF> {
        let start = S::offset_of::<D>().unwrap();

        // # Safety:
        //
        // `<S as DoF>::DoF` implements DimName, which means `RStride` always equal to `Const<1>`,
        // and `CStride` equal to `S::DoF`.
        unsafe {
            let data = ViewStorage::new_with_strides_unchecked(
                &self.data,
                (start, 0),
                (D::DoF::name(), S::DoF::name()),
                (Const::<1>, S::DoF::name()),
            );
            Matrix::from_data_statically_unchecked(data)
        }
    }
}
