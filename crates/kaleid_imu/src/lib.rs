use derive_deref::{Deref, DerefMut};
pub use kaleid::DoF;
use kaleid::{
    Const, Element,
    error_state::{ErrorState, Velocity},
    extract::{Access, Extract},
    predict::{BuildTransition, TransitionViewMut},
};
use nalgebra::{Rotation3, Vector3};

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Acceleration(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2, I3, I4> BuildTransition<S, (I1, I2, I3, I4)> for Acceleration
where
    for<'a> &'a S: Extract<(&'a Rotation3<Element>, &'a Self), I1>,
    S: Access<Velocity, I2> + Access<Rotation3<Element>, I3> + Access<Self, I4>,
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
    S: Access<Velocity, I1> + Access<Self, I2>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, _: &S, dt: Element) {
        transition.block::<Velocity, Self, _, _>().fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Gyro(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2, I3> BuildTransition<S, (I1, I2, I3)> for Gyro
where
    for<'a> &'a S: Access<&'a Self, I1>,
    S: Access<Rotation3<Element>, I2> + Access<Self, I3>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let gyro = states.pluck().0;
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
