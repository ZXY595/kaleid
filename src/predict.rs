use derive_deref::{Deref, DerefMut};
pub use nalgebra::Const;
use nalgebra::{
    DefaultAllocator, DimName, Matrix, MatrixViewMut, OMatrix, RawStorageMut as _, Storage,
    ViewStorageMut, allocator::Allocator,
};

use crate::{Covariance, Element, error_state::ErrorState};

pub trait BuildTransition<S: ErrorState> {
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let _ = (transition, states, dt);
    }
}

/// A mutable view of the transition matrix
#[derive(Deref, DerefMut)]
pub struct TransitionViewMut<'a, D: ErrorState>(MatrixViewMut<'a, Element, D::DoF, D::DoF>);

impl<'a, D: ErrorState> TransitionViewMut<'a, D> {
    pub fn from_matrix_mut(matrix: &mut OMatrix<Element, D::DoF, D::DoF>) -> Self
    where
        DefaultAllocator: Allocator<D::DoF, D::DoF>,
    {
        let (row, col) = matrix.shape_generic();

        // # Safety:
        //
        // `<D as DoF>::DoF` implements DimName, which means `RStride` always equal to `Const<1>`,
        // and `CStride` equal to `D::DoF`.
        let matrix = unsafe {
            let data = ViewStorageMut::from_raw_parts(
                matrix.data.ptr_mut(),
                (row, col),
                (Const::<1>, row),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        Self(matrix)
    }

    pub fn block<D1: ErrorState, D2: ErrorState>(
        &mut self,
    ) -> MatrixViewMut<'_, Element, D1::DoF, D2::DoF, Const<1>, D::DoF>
    where
        D: AsRef<D1> + AsRef<D2>,
    {
        self.generic_view_mut(
            (D::offset::<D1>(), D::offset::<D2>()),
            (D1::DoF::name(), D2::DoF::name()),
        )
    }

    pub fn set_block<D1: ErrorState, D2: ErrorState>(
        &mut self,
        block: &Matrix<Element, D1::DoF, D2::DoF, impl Storage<Element, D1::DoF, D2::DoF>>,
    ) -> &mut Self
    where
        D: AsRef<D1> + AsRef<D2>,
    {
        self.block::<D1, D2>().copy_from(block);
        self
    }

    pub fn set_self_block<DD: ErrorState>(
        &mut self,
        block: &Matrix<Element, DD::DoF, DD::DoF, impl Storage<Element, DD::DoF, DD::DoF>>,
    ) -> &mut Self
    where
        D: AsRef<DD>,
    {
        self.set_block::<DD, DD>(block)
    }

    pub fn set_block_with<D1: ErrorState, D2: ErrorState>(
        &mut self,
        f: impl FnOnce(MatrixViewMut<'_, Element, D1::DoF, D2::DoF, Const<1>, D::DoF>),
    ) -> &mut Self
    where
        D: AsRef<D1> + AsRef<D2>,
    {
        f(self.block::<D1, D2>());
        self
    }
}

impl<D: ErrorState> Covariance<D>
where
    DefaultAllocator: Allocator<D::DoF, D::DoF> + Allocator<D::DoF>,
{
    pub fn predict(
        &mut self,
        states: &D,
        transition: &mut OMatrix<Element, D::DoF, D::DoF>,
        noise: &OMatrix<Element, D::DoF, D::DoF>,
        dt: Element,
    ) where
        D: BuildTransition<D>,
    {
        D::build_transition(
            &mut TransitionViewMut::from_matrix_mut(transition),
            states,
            dt,
        );
        let mut temp = noise * dt.powi(2);
        temp.quadform_tr(1.0, transition, self, 1.0);
        self.0 = temp;
    }
}

// impl<S: DoF, T1, T2, I1, I2> BuildTransition<S, (I1, I2)> for (T1, T2)
// where
//     T1: BuildTransition<S, I1>,
//     T2: BuildTransition<S, I2>,
// {
//     fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
//         T1::build_transition(transition, states, dt);
//         T2::build_transition(transition, states, dt);
//     }
// }
