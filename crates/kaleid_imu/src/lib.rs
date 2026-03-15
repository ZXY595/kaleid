use derive_deref::{Deref, DerefMut};
pub use kaleid::DoF;
use kaleid::{
    Const, Element,
    error_state::{ErrorState, Velocity},
    extract::{Extract, Pluck},
    predict::{BuildTransition, TransitionViewMut},
};
use nalgebra::{Rotation3, Storage, UnitQuaternion, Vector, Vector3};

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Acceleration(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2, I3, I4> BuildTransition<S, (I1, I2, I3, I4)> for Acceleration
where
    for<'a> &'a S: Extract<(&'a UnitQuaternion<Element>, &'a Self), I1>,
    S: Pluck<Velocity, I2, Before: DoF>
        + Pluck<UnitQuaternion<Element>, I3, Before: DoF>
        + Pluck<Self, I4, Before: DoF>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let (rot, acc) = states.extract().0;
        let rot = rot.to_rotation_matrix();
        transition
            .set_block::<Velocity, UnitQuaternion<Element>, _, _>(&(rot * acc.cross_matrix() * -dt))
            .set_block::<Velocity, Self, _, _>(&(rot.matrix() * dt));
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct AccelerationBias(#[DoF = 3] Vector3<Element>);

#[derive(Debug, Deref, DerefMut)]
pub struct Gravity(Vector3<Element>);

impl DoF for Gravity {
    type DoF = Const<3>;
}

impl ErrorState for Gravity {
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        self.0.inject(inj)
    }
}

impl<S: DoF, I1, I2> BuildTransition<S, (I1, I2)> for Gravity
where
    S: Pluck<Velocity, I1, Before: DoF> + Pluck<Self, I2, Before: DoF>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, _: &S, dt: Element) {
        transition.block::<Velocity, Self, _, _>().fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Gyro(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2, I3> BuildTransition<S, (I1, I2, I3)> for Gyro
where
    for<'a> &'a S: Pluck<&'a Self, I1>,
    S: Pluck<UnitQuaternion<Element>, I2, Before: DoF> + Pluck<Self, I3, Before: DoF>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, states: &S, dt: Element) {
        let gyro = states.pluck().0;
        transition
            .set_self_block::<UnitQuaternion<Element>, _>(
                Rotation3::from_scaled_axis(gyro.0 * -dt).matrix(),
            )
            .block::<UnitQuaternion<Element>, Self, _, _>()
            .fill_diagonal(dt);
    }
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct GyroBias(#[DoF = 3] Vector3<Element>);
