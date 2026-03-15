use nalgebra::{RealField, stack};

use crate::{
    algorithm::{
        imu::measure::{ImuMeasured, ImuObserved, StampedImu},
        lio::{LIO, state::State},
    },
    utils::ToRadians,
};
use kaleid::Eskf;

// impl<T> Eskf<State<T>>
// where
//     T: RealField + ToRadians,
// {
//     fn observe_imu(
//         &self,
//         gravity_factor: T,
//         measure_noise: &ImuMeasured<T>,
//         imu_acc: &ImuMeasured<T>,
//     ) -> ImuObserved<T> {
//         let measured_linear_acc = imu_acc.acc * gravity_factor - self.state.acc.biased().deref();
//
//         let measured_angular_acc = imu_acc.gyro - self.state.gyro.biased().deref();
//
//         #[expect(clippy::toplevel_ref_arg)]
//         let measurement = stack![measured_linear_acc; measured_angular_acc];
//
//         #[expect(clippy::toplevel_ref_arg)]
//         let noise = stack![measure_noise.acc; measure_noise.gyro];
//
//         ImuObserved::new_no_model(measurement, noise)
//     }
// }

impl<T> Extend<StampedImu<T>> for LIO<T>
where
    T: RealField + ToRadians,
{
    fn extend<I>(&mut self, imus: I)
    where
        I: IntoIterator<Item = StampedImu<T>>,
    {
        imus.into_iter().for_each(|imu| {
            self.eskf.update(imu.timestamp, |eskf| {
                (eskf.observe_imu(
                    self.gravity_factor.clone(),
                    &self.measure_noise.imu_acc,
                    &imu.measured,
                ),)
            });
        })
    }
}
