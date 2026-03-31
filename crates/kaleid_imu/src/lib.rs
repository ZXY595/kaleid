use derive_deref::{Deref, DerefMut};
use extension_traits::extension;
use kaleid::{
    Covariance, Element,
    error_state::{ErrorState, Velocity},
    predict::{BuildTransition, TransitionViewMut},
};
use nalgebra::{DefaultAllocator, Rotation3, Vector3, Vector6, allocator::Allocator};

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Acceleration(#[DoF = 3] Vector3<Element>);

impl<S: ErrorState> BuildTransition<S> for Acceleration
where
    S: AsRef<Velocity> + AsRef<Rotation3<Element>> + AsRef<Self>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let rot: &Rotation3<Element> = states.as_ref();
        let acc: &Self = states.as_ref();
        transition
            .set_block::<Velocity, Rotation3<Element>>(&(rot * acc.cross_matrix() * -dt))
            .set_block::<Velocity, Self>(&(rot.matrix() * dt));
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct AccelerationBias(#[DoF = 3] Vector3<Element>);

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Gravity(#[DoF = 3] Vector3<Element>);

impl<S: ErrorState> BuildTransition<S> for Gravity
where
    S: AsRef<Velocity> + AsRef<Self>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, _: &S, dt: Element) {
        transition.block::<Velocity, Self>().fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Gyro(#[DoF = 3] Vector3<Element>);

impl<S: ErrorState> BuildTransition<S> for Gyro
where
    S: AsRef<Rotation3<Element>> + AsRef<Self>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let gyro: &Self = states.as_ref();
        transition
            .set_self_block::<Rotation3<Element>>(
                Rotation3::from_scaled_axis(gyro.0 * -dt).matrix(),
            )
            .block::<Rotation3<Element>, Self>()
            .fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct GyroBias(#[DoF = 3] Vector3<Element>);

#[derive(ErrorState)]
struct ImuMeasure {
    accel: Acceleration,
    gyro: Gyro,
}

#[derive(ErrorState)]
struct ImuMeasureBias {
    accel_bias: AccelerationBias,
    gyro_bias: GyroBias,
}

#[expect(private_bounds)]
#[extension(pub trait ObserveImu)]
impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF>
        + Allocator<S::DoF>
        + Allocator<<ImuMeasure as ErrorState>::DoF, S::DoF>
        + Allocator<S::DoF, <ImuMeasure as ErrorState>::DoF>,
{
    fn observe_imu(
        &mut self,
        states: &mut S,
        accel: &Vector3<Element>,
        gyro: &Vector3<Element>,
        measure_noise: &Vector6<Element>,
        gravity_factor: Element,
    ) where
        S: AsRef<Acceleration> + AsRef<Gyro> + AsRef<AccelerationBias> + AsRef<GyroBias>,
    {
        let accel_bias: &AccelerationBias = states.as_ref();
        let gyro_bias: &GyroBias = states.as_ref();

        let accel = accel * gravity_factor - accel_bias.0;
        let gyro = gyro - gyro_bias.0;
        let measurement = Vector6::new(accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z);

        let cross_cov_tr = self.rows::<ImuMeasure>() + self.rows::<ImuMeasureBias>();

        let innovation_cov = self.block::<ImuMeasure, ImuMeasure>()
            + self.block::<ImuMeasure, ImuMeasureBias>()
            + self.block::<ImuMeasureBias, ImuMeasure>();

        self.observe::<<ImuMeasure as ErrorState>::DoF>(
            states,
            &measurement,
            measure_noise,
            &cross_cov_tr,
            innovation_cov,
        )
    }
}
