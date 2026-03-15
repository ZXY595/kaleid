mod init;

use kaleid::{
    State,
    observe::{NoModelObservation, StampedMeasurement},
    state::{KFState, StateDim},
};
use std::ops::Deref;

use nalgebra::{ClosedAddAssign, RealField, Scalar, stack};
use num_traits::Zero;
use odometries_macros::KFState;

use super::state::{AccState, GyroState, ImuState};
use crate::{algorithm::imu::state::Accel, utils::ToRadians};

pub use init::ImuInit;

pub type ImuObserved<T> = NoModelObservation<AccState<T>, ImuState<T>, StateDim<Accel<T>>>;

#[derive(KFState)]
#[element(T: Scalar + ClosedAddAssign)]
pub struct ImuMeasured<T> {
    acc: AccState<T>,
    gyro: GyroState<T>,
}

pub type StampedImu<T> = StampedMeasurement<T, ImuMeasured<T>>;

impl<T> ImuState<T>
where
    T: RealField + ToRadians,
{
    fn observe_imu(
        &self,
        gravity_factor: T,
        measure_noise: &ImuMeasured<T>,
        imu_acc: &ImuMeasured<T>,
    ) -> ImuObserved<T> {
        let measured_linear_acc = imu_acc.acc.0 * gravity_factor - self.acc.unbias().0;

        let measured_angular_acc = imu_acc.gyro.0 - self.gyro.unbias().0;

        #[expect(clippy::toplevel_ref_arg)]
        let measurement = stack![measured_linear_acc; measured_angular_acc];

        #[expect(clippy::toplevel_ref_arg)]
        let noise = stack![measure_noise.acc.0; measure_noise.gyro.0];

        ImuObserved::new_no_model(measurement, noise)
    }
}

impl<T> Default for ImuMeasured<T>
where
    T: Scalar + Zero,
{
    fn default() -> Self {
        Self {
            acc: AccState::default(),
            gyro: GyroState::default(),
        }
    }
}
