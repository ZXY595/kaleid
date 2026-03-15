//! Most of ideas comes from Leg-Kilo

pub mod config;
pub mod downsample;
pub mod measurement;
pub mod predict;
pub mod state;

use std::ops::Deref;

use state::State;

use crate::{
    algorithm::imu::{
        measure::ImuInit,
        state::{AccState, GravityState, VelocityState},
    },
    frame::{IsometryFramed, frames},
    utils::ToRadians,
    voxel_map::{VoxelMap, uncertain::plane::Plane},
};
pub use config::{BodyPointProcessCov, Config, NoGravityConfig};
use downsample::{Downsampler, ScanDownsampler};
use kaleid::{Eskf, state::common::PoseState};
use measurement::ProcessingPoints;

use nalgebra::RealField;

pub use measurement::MeasureNoiseConfig;

pub struct LIO<T>
where
    T: RealField,
{
    eskf: Eskf<State<T>>,
    map: VoxelMap<T>,
    downsampler: ScanDownsampler<T>,
    points_process_buffer: Vec<ProcessingPoints<T>>,
    // configs
    body_point_process_cov: BodyPointProcessCov<T>,
    measure_noise: MeasureNoiseConfig<T>,
    extrinsics: IsometryFramed<T, fn(frames::BodyFrame) -> frames::ImuFrame>,
    gravity_factor: T,
}

impl<T> ImuInit<T>
where
    T: RealField + ToRadians,
{
    pub fn new_lio(self, lio_config: Config<T>) -> LIO<T> {
        LIO::new(lio_config, self)
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians,
{
    pub fn new(config: Config<T>, imu_init: ImuInit<T>) -> Self {
        let (gravity, config) = config.take_gravity();
        let gravity_factor = gravity / imu_init.linear_acc_norm.clone();

        let mut lio =
            Self::new_with_gravity_factor(config, imu_init.timestamp_init, gravity_factor.clone());

        let gravity = imu_init.linear_acc_mean.deref() * gravity_factor;

        let imu = &mut lio.eskf.imu;
        imu.acc.state = gravity.clone();
        imu.gravity = GravityState::new(gravity);
        imu.gyro.bias = imu_init.angular_acc_bias;

        lio
    }

    /// Create a new LIO instance with a given gravity factor.
    ///
    /// This does not need the `gravity` in [`Config<T>`], provide [`NoGravityConfig<T>`] instead.
    pub fn new_with_gravity_factor(
        config: NoGravityConfig<T>,
        timestamp_init: T,
        gravity_factor: T,
    ) -> Self {
        let eskf = Eskf::new(config.process_cov.state.into(), timestamp_init);

        Self {
            eskf,
            map: VoxelMap::new(config.voxel_map),
            downsampler: Downsampler::new(config.downsample_resolution),
            points_process_buffer: Vec::with_capacity(config.buffer_init_size),
            body_point_process_cov: config.process_cov.body_point,
            measure_noise: config.measure_noise,
            extrinsics: config.extrinsics,
            gravity_factor,
        }
    }

    #[inline]
    pub fn get_pose(&self) -> &PoseState<T> {
        &self.eskf.pose
    }

    #[inline]
    pub fn get_velocity(&self) -> &VelocityState<T> {
        &self.eskf.imu.velocity
    }

    #[inline]
    pub fn planes(&self) -> impl Iterator<Item = &Plane<T>> {
        self.map.planes()
    }
}
