use extension_traits::extension;
use nalgebra::{
    Cholesky, Const, DefaultAllocator, DimName, Matrix, MatrixView, OMatrix, Storage, Vector,
    ViewStorage, allocator::Allocator,
};

use crate::{
    Covariance, DoF, Element,
    error_state::ErrorState,
    extract::{Extract, ExtractMode},
};

impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF> + Allocator<S::DoF>,
{
    pub fn observe<D: DoF>(
        &mut self,
        states: &mut S,
        measurement: &Vector<Element, D::DoF, impl Storage<Element, D::DoF>>,
        noise: &Vector<Element, D::DoF, impl Storage<Element, D::DoF>>,
        cross_cov_tr: &Matrix<Element, D::DoF, S::DoF, impl Storage<Element, D::DoF, S::DoF>>,
        innovation_cov: Matrix<Element, D::DoF, D::DoF, impl Storage<Element, D::DoF, D::DoF>>,
    ) where
        DefaultAllocator:
            Allocator<D::DoF, D::DoF> + Allocator<S::DoF, D::DoF> + Allocator<D::DoF, S::DoF>,
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
        diagnoal: &Vector<Element, D, impl Storage<Element, D>>,
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

impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF>,
{
    /// If `M1` or `M2` is [`Multi`](crate::extract::Multi), you must ensure that
    /// `D1` or `D2` is contiguous, otherwise you will get a compile time panic
    pub fn block<D1: DoF, D2: DoF, M1: ExtractMode, M2: ExtractMode, I1, I2>(
        &self,
    ) -> MatrixView<'_, Element, D1::DoF, D2::DoF, Const<1>, S::DoF>
    where
        S: Extract<D1, I1, M1> + Extract<D2, I2, M2>,
    {
        let start = <S as Extract<D1, I1, M1>>::OFFSET
            .into()
            .zip(<S as Extract<D2, I2, M2>>::OFFSET.into())
            .expect("states is not contiguous");

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

    /// If `M`  is [`Multi`](crate::extract::Multi), you must ensure that
    /// `D` is contiguous, otherwise you will get a compile time panic
    pub fn rows<D: DoF, M: ExtractMode, I>(
        &self,
    ) -> MatrixView<'_, Element, D::DoF, S::DoF, Const<1>, S::DoF>
    where
        S: Extract<D, I, M>,
    {
        let start = S::OFFSET.into().expect("states is not contiguous");

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
