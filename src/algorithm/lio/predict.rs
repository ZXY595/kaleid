use std::ops::{Deref, DerefMut};

use super::State;
use crate::algorithm::lio::LIO;
use kaleid::{ImplCov, Covariance, DeltaTime, Eskf, KFState, StatePredict, StateOffset};

use nalgebra::{IsometryMatrix3, RealField, Rotation3, SMatrix, Scalar, Translation3};
use num_traits::Zero;
use simba::scalar::SupersetOf;

pub struct ProcessCovConfig<T> {
    pub velocity: T,
    pub gyro: T,
    pub gyro_bias: T,
    pub accel: T,
    pub accel_bias: T,
}

impl<T: SupersetOf<f64>> Default for ProcessCovConfig<T> {
    fn default() -> Self {
        Self {
            velocity: nalgebra::convert(20.0),
            accel: nalgebra::convert(500.0),
            accel_bias: nalgebra::convert(0.0001),
            gyro: nalgebra::convert(1000.0),
            gyro_bias: nalgebra::convert(0.0001),
        }
    }
}

impl<T> StatePredict for State<T>
where
    T: RealField,
{
    fn predict(&mut self, dt: T) {
        let imu = &self.imu;

        let delta_rotation = Rotation3::new(imu.gyro.deref() * dt.clone());
        let delta_translation = Translation3::from(imu.velocity.deref() * dt.clone());
        let delta_velocity = (self.pose.deref() * imu.acc.deref() - imu.gravity.deref()) * dt;

        *self.pose *= delta_rotation;
        *self.pose *= delta_translation;
        *self.imu.velocity += delta_velocity;
    }

    fn predict_cov<Super: KFState>(&self, fx: &mut ImplCov!(Super), dt: T)
    where
        Self: StateOffset<Super>,
    {
        todo!()
    }
}

// impl<T> Eskf<State<T>>
// where
//     T: RealField,
// {
//     pub fn predict_cov(&mut self, dt: T) {
//         let state = &self.state;
//
//         let mut fx = Covariance::<State<T>>(SMatrix::identity());
//
//         fx.sub_cov::<RotationState<T>>()
//             .copy_from(Rotation3::new(state.gyro.deref() * -dt.clone()).matrix());
//
//         fx.get_mut::<RotationState<T>, VelocityState<T>>()
//             .copy_from(&(&state.pose.rotation * state.acc.cross_matrix() * -dt.clone()));
//
//         fx.get_mut::<VelocityState<T>, PositionState<T>>()
//             .fill_diagonal(dt.clone());
//
//         fx.get_mut::<GravityState<T>, VelocityState<T>>()
//             .fill_diagonal(dt.clone());
//
//         fx.get_mut::<AccState<T>, VelocityState<T>>()
//             .copy_from(&(state.pose.rotation.matrix() * dt.clone()));
//
//         fx.get_mut::<GyroState<T>, RotationState<T>>()
//             .fill_diagonal(dt.clone());
//
//         // X = Fx.X.FX' + Q * dt^2
//         let mut cov = self.process_cov.deref() * dt.powi(2);
//         cov.quadform_tr(T::one(), &fx, self.cov.deref(), T::one());
//         *self.cov = cov;
//     }
// }

// impl<T: RealField> StatePredict<DeltaTime<T>> for Eskf<State<T>> {
//     fn predict(&mut self, dt: DeltaTime<T>) {
//         self.state.predict(dt.predict);
//         self.predict_cov(dt.observe);
//     }
// }

impl<T: RealField> From<ProcessCovConfig<T>> for Covariance<State<T>> {
    fn from(value: ProcessCovConfig<T>) -> Self {
        let mut cov = Self::default();

        cov.sub_cov_mut::<VelocityState<T>>()
            .fill_diagonal(value.velocity);

        cov.sub_cov_mut::<AccState<T>>().fill_diagonal(value.accel);

        cov.sub_cov_mut::<AccBiasState<T>>()
            .fill_diagonal(value.accel_bias);

        cov.sub_cov_mut::<GyroState<T>>().fill_diagonal(value.gyro);

        cov.sub_cov_mut::<GyroBiasState<T>>()
            .fill_diagonal(value.gyro_bias);

        cov
    }
}
