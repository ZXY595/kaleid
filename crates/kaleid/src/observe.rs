mod diagonal_view;
mod substitute;

use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, Matrix, Owned, Storage, allocator::Allocator,
};

use crate::{
    ImplMatrix, ImplVector,
    covariance::{CovColsView, CovRowsView, CovView, FromCov, MembersOf},
    observe::diagonal_view::SquareDiagDim,
    state::bundle::StateBundle,
};
use diagonal_view::ViewDiagonalMut;
use substitute::InverseWithSubstitute;

use super::Eskf;

pub struct JointCov<Bundle: StateBundle, D, S1, S2, S3> {
    pub cross_cov: Matrix<Bundle::Element, Bundle::Dim, D, S1>,
    pub cross_cov_transpose: Matrix<Bundle::Element, D, Bundle::Dim, S2>,
    pub innovation_cov: Matrix<Bundle::Element, D, D, S3>,
}

impl<Bundle, const BUNDLE_DIM: usize> Eskf<Bundle>
where
    Bundle: StateBundle<Element: ComplexField, Dim = Const<BUNDLE_DIM>>,
{
    pub fn observe_() {}
    pub fn observe<State, const STATE_DIM: usize, D: SquareDiagDim, FC, S1, S2, S3>(
        &mut self,
        measurement: ImplVector!(Bundle::Element, D),
        noise: ImplVector!(Bundle::Element, D),
        model: impl FnOnce(FC) -> JointCov<Bundle, D, S1, S2, S3>,
    ) where
        State: MembersOf<Bundle, Dim = Const<STATE_DIM>>,
        FC: for<'cov> FromCov<'cov, Bundle>,
        S1: Storage<Bundle::Element, Bundle::Dim, D>,
        S2: Storage<Bundle::Element, D, Bundle::Dim>,
        S3: Storage<Bundle::Element, D, D>,
        DefaultAllocator:
            Allocator<D, D> + Allocator<D> + Allocator<Bundle::Dim, D> + Allocator<D, Bundle::Dim>,
    {
        let JointCov {
            cross_cov,
            cross_cov_transpose,
            innovation_cov,
        } = model(FromCov::from_cov(&mut self.cov.as_view_mut()));

        let kalman_gain = cross_cov
            * innovation_cov
                .into_owned()
                .diagonal_add(noise)
                .cholesky_inverse_with_substitute();

        self.state.gain(&kalman_gain * measurement);
        self.cov -= kalman_gain * cross_cov_transpose;
    }
}

impl<Bundle: StateBundle, D, S1, S2, S3> JointCov<Bundle, D, S1, S2, S3> {
    pub fn new(
        cross_cov: Matrix<Bundle::Element, Bundle::Dim, D, S1>,
        cross_cov_transpose: Matrix<Bundle::Element, D, Bundle::Dim, S2>,
        innovation_cov: Matrix<Bundle::Element, D, D, S3>,
    ) -> Self {
        Self {
            cross_cov,
            cross_cov_transpose,
            innovation_cov,
        }
    }
}

type QueryCov<'cov, Bundle, State> = (
    CovColsView<'cov, Bundle, State>,
    CovRowsView<'cov, Bundle, State>,
    CovView<'cov, Bundle, State>,
);

impl<Bundle, const B_DIM: usize, D: Dim>
    JointCov<
        Bundle,
        D,
        Owned<Bundle::Element, Bundle::Dim, D>,
        Owned<Bundle::Element, D, Bundle::Dim>,
        Owned<Bundle::Element, D, D>,
    >
where
    Bundle: StateBundle<Element: ComplexField, Dim = Const<B_DIM>>,
    DefaultAllocator: Allocator<Bundle::Dim, D> + Allocator<D, Bundle::Dim> + Allocator<D, D>,
{
    fn from_matrix<State: MembersOf<Bundle, Dim = Const<S_DIM>>, const S_DIM: usize>(
        model: ImplMatrix!(Bundle::Element, State::Dim, D),
    ) -> impl FnOnce(QueryCov<'_, Bundle, State>) -> Self
    where
        DefaultAllocator: Allocator<D, State::Dim>,
    {
        move |(cols, rows, s_cov)| {
            let cross_cov = cols.0 * &model;
            let innovation_cov = model.transpose() * s_cov.0 * &model;
            JointCov {
                cross_cov,
                cross_cov_transpose: model.transpose() * rows.0,
                innovation_cov,
            }
        }
    }
}
