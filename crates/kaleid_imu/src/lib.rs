use derive_deref::{Deref, DerefMut};
use extension_traits::extension;
pub use kaleid::DoF;
use kaleid::{
    Const, Covariance, Element,
    error_state::{ErrorState, Velocity},
    extract::{Extract, Multi},
    predict::{BuildTransition, TransitionViewMut},
};
use nalgebra::{DefaultAllocator, Rotation3, Vector3, Vector6, allocator::Allocator};

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Acceleration(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2, I3, I4> BuildTransition<S, (I1, I2, I3, I4)> for Acceleration
where
    for<'a> &'a S: Extract<(&'a Rotation3<Element>, &'a Self), I1, Multi>,
    S: Extract<Velocity, I2> + Extract<Rotation3<Element>, I3> + Extract<Self, I4>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let (rot, acc) = states.extract().0;
        transition
            .set_block::<Velocity, Rotation3<Element>, _, _>(&(rot * acc.cross_matrix() * -dt))
            .set_block::<Velocity, Self, _, _>(&(rot.matrix() * dt));
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct AccelerationBias(#[DoF = 3] Vector3<Element>);

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Gravity(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2> BuildTransition<S, (I1, I2)> for Gravity
where
    S: Extract<Velocity, I1> + Extract<Self, I2>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, _: &S, dt: Element) {
        transition.block::<Velocity, Self, _, _>().fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Gyro(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2, I3> BuildTransition<S, (I1, I2, I3)> for Gyro
where
    for<'a> &'a S: Extract<&'a Self, I1>,
    S: Extract<Rotation3<Element>, I2> + Extract<Self, I3>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let gyro = states.extract().0;
        transition
            .set_self_block::<Rotation3<Element>, _>(
                Rotation3::from_scaled_axis(gyro.0 * -dt).matrix(),
            )
            .block::<Rotation3<Element>, Self, _, _>()
            .fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct GyroBias(#[DoF = 3] Vector3<Element>);

#[extension(pub trait ObserveImu)]
impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF>
        + Allocator<S::DoF>
        + Allocator<Const<6>, S::DoF>
        + Allocator<S::DoF, Const<6>>,
{
    fn observe_imu<I1, I2, I3>(
        &mut self,
        states: &mut S,
        accel: &Vector3<Element>,
        gyro: &Vector3<Element>,
        measure_noise: &Vector6<Element>,
        gravity_factor: Element,
    ) where
        for<'a> &'a mut S: Extract<(&'a AccelerationBias, &'a GyroBias), I1, Multi>,
        S: Extract<(Acceleration, Gyro), I2, Multi>
            + Extract<(AccelerationBias, GyroBias), I3, Multi>,
    {
        let (accel_bias, gyro_bias) = states.extract().0;

        let accel = accel * gravity_factor - accel_bias.0;
        let gyro = gyro - gyro_bias.0;
        let measurement = Vector6::new(accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z);

        let cross_cov_tr = self.rows::<(Acceleration, Gyro), _, _>()
            + self.rows::<(AccelerationBias, GyroBias), _, _>();
        let innovation_cov = self.block::<(Acceleration, Gyro), (Acceleration, Gyro), _, _, _, _>()
            + self.block::<(Acceleration, Gyro), (AccelerationBias, GyroBias), _, _, _, _>()
            + self.block::<(AccelerationBias, GyroBias), (Acceleration, Gyro), _, _, _, _>();

        self.observe::<(Acceleration, Gyro)>(
            states,
            &measurement,
            measure_noise,
            &cross_cov_tr,
            innovation_cov,
        )
    }
}
