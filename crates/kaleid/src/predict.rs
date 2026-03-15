use nalgebra::{ComplexField, DefaultAllocator, Matrix, MatrixViewMut, allocator::Allocator};

use crate::KFState;
use num_traits::One;

use super::Eskf;

pub trait PredictCov<S: KFState> {
    fn predict_cov(fx: &mut MatrixViewMut<'_, S::Element, S::Dim, S::Dim>);
}

impl<S> Eskf<S>
where
    S: KFState<Element: ComplexField>,
    DefaultAllocator: Allocator<S::Dim, S::Dim> + Allocator<S::Dim>,
{
    pub fn predict_cov(&mut self, dt: S::Element) {
        let mut fx = Matrix::identity();
        todo!();
        // self.bundle.predict_cov(dt.clone(), &mut fx);

        // X = Fx.X.FX' + dt^2 * Q
        let mut temp = &self.process_cov * dt.clone() * dt;
        temp.quadform_tr(S::Element::one(), &fx, &self.cov, S::Element::one());
        self.cov = temp;
    }
}
