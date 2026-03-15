use kaleid::state::common::PoseState;
use odometries_macros::KFState;

use crate::algorithm::imu::state::ImuState;

use nalgebra::RealField;

#[derive(KFState)]
#[element(T: RealField)]
pub struct State<T> {
    /// The pose of the body in the world frame.
    /// use to transform from body to world frame.
    pub pose: PoseState<T>,

    pub imu: ImuState<T>,
}

impl<T> Default for State<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            pose: Default::default(),
            imu: Default::default(),
        }
    }
}
