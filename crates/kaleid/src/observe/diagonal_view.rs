use crate::ImplVector;
use nalgebra::{
    ClosedAddAssign, DefaultAllocator, Dim, DimAdd, DimMin, DimMinimum, DimSum, Matrix, OMatrix,
    RawStorageMut as _, Scalar, U1, VectorViewMut, ViewStorageMut, allocator::Allocator,
};

pub(crate) trait SquareDiagDim: DimMin<Self, Output = Self> + DimAdd<U1> {}

pub trait ViewDiagonalMut {
    type Element;
    type Dim: Dim;
    type RStride: Dim;

    fn view_diagonal_mut(
        &mut self,
    ) -> VectorViewMut<'_, Self::Element, Self::Dim, Self::RStride, U1>;

    #[inline]
    fn diagonal_add(mut self, value: ImplVector!(Self::Element, Self::Dim)) -> Self
    where
        Self: Sized,
        Self::Element: Scalar + ClosedAddAssign,
    {
        use std::ops::AddAssign;
        self.view_diagonal_mut().add_assign(value);
        self
    }
}

impl<T, R: Dim, C: Dim> ViewDiagonalMut for OMatrix<T, R, C>
where
    T: Scalar,
    R: DimMin<C> + DimAdd<U1>,
    DefaultAllocator: Allocator<R, C>,
{
    type Element = T;
    type Dim = DimMinimum<R, C>;
    type RStride = DimSum<R, U1>;
    #[inline]
    fn view_diagonal_mut(
        &mut self,
    ) -> VectorViewMut<'_, Self::Element, Self::Dim, Self::RStride, U1> {
        let (rows, cols) = self.shape_generic();
        let min_dim = rows.min(cols);
        // # SAFETY:
        //
        // The data is guaranteed to be contiguous and in row-major order.
        unsafe {
            let data = ViewStorageMut::from_raw_parts(
                self.data.ptr_mut(),
                (min_dim, U1),
                (rows.add(U1), U1),
            );
            Matrix::from_data_statically_unchecked(data)
        }
    }
}

impl<D> SquareDiagDim for D where D: DimMin<Self, Output = Self> + DimAdd<U1> {}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::*;

    #[test]
    fn test_diagonal_view() {
        let mut square_m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];
        square_m.view_diagonal_mut().fill(0.0);
        assert_eq!(
            square_m,
            matrix![
                0.0, 2.0, 3.0;
                4.0, 0.0, 6.0;
                7.0, 8.0, 0.0
            ]
        );

        let mut rect_m = matrix![
            1.0,  2.0,  3.0,  4.0,  5.0;
            6.0,  7.0,  8.0,  9.0,  10.0;
            11.0, 12.0, 13.0, 14.0, 15.0;
        ];
        rect_m.view_diagonal_mut().fill(0.0);
        assert_eq!(
            rect_m,
            matrix![
                0.0,  2.0,  3.0,  4.0,  5.0;
                6.0,  0.0,  8.0,  9.0,  10.0;
                11.0, 12.0, 0.0,  14.0, 15.0;
            ]
        );

        let mut rect_m_tr = matrix![
            1.0,  6.0,  11.0;
            2.0,  7.0,  12.0;
            3.0,  8.0,  13.0;
            4.0,  9.0,  14.0;
            5.0,  10.0, 15.0;
        ];
        rect_m_tr.view_diagonal_mut().fill(0.0);
        assert_eq!(
            rect_m_tr,
            matrix![
                0.0,  6.0,  11.0;
                2.0,  0.0,  12.0;
                3.0,  8.0,  0.0;
                4.0,  9.0,  14.0;
                5.0,  10.0, 15.0;
            ]
        );

        // test dyn dim matrix
        let mut rect_m_dyn = rect_m.resize(3, 5, 16.0);
        rect_m_dyn.view_diagonal_mut().fill(1.0);
        assert_eq!(
            rect_m_dyn,
            matrix![
                1.0,  2.0,  3.0,  4.0,  5.0;
                6.0,  1.0,  8.0,  9.0,  10.0;
                11.0, 12.0, 1.0,  14.0, 15.0;
            ]
        );

        let mut rect_m_tr_dyn = rect_m_tr.resize(5, 3, 16.0);
        rect_m_tr_dyn.view_diagonal_mut().fill(1.0);
        assert_eq!(
            rect_m_tr_dyn,
            matrix![
                1.0,  6.0,  11.0;
                2.0,  1.0,  12.0;
                3.0,  8.0,  1.0;
                4.0,  9.0,  14.0;
                5.0,  10.0, 15.0;
            ]
        );
    }
}
