use kaleid::{ImplCov, KFState, StatePredict, StateOffset};

use super::state::ImuState;

impl<T> StatePredict for ImuState<T> {
    fn predict(&mut self, dt: T) {
        self.velocity += self.acc * dt;
    }

    fn predict_cov(&self, dt: Self::Element, fx: &mut ImplCov!(Self)) {
        todo!()
    }
}
