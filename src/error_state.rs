use crate::{DoF, Element};

use derive_deref::{Deref, DerefMut};
pub use nalgebra::Const;
use nalgebra::{
    DimName, Rotation3, SVector, Storage, Translation3, UnitQuaternion, Vector, Vector3,
};

use crate::{
    extract::Extract,
    predict::{BuildTransition, TransitionViewMut},
};

pub trait ErrorState: DoF {
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>);
}
pub use kaleid_macros::ErrorState;

impl<const D: usize> DoF for SVector<Element, D> {
    type DoF = Const<D>;
}

impl<const D: usize> ErrorState for SVector<Element, D> {
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self += inj
    }
}

impl DoF for Translation3<Element> {
    type DoF = Const<3>;
}

impl ErrorState for Translation3<Element> {
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self *= Translation3::from(inj.into_owned())
    }
}

impl DoF for UnitQuaternion<Element> {
    type DoF = Const<3>;
}

impl ErrorState for UnitQuaternion<Element> {
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self *= UnitQuaternion::from_scaled_axis(inj.into_owned())
    }
}

impl DoF for Rotation3<Element> {
    type DoF = Const<3>;
}

impl ErrorState for Rotation3<Element> {
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self *= Rotation3::from_scaled_axis(inj.into_owned())
    }
}

mod kaleid {
    pub use crate::*;
}

#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Velocity(#[DoF = 3] Vector3<Element>);

impl<S: DoF, I1, I2> BuildTransition<S, (I1, I2)> for Velocity
where
    S: Extract<Translation3<Element>, I1> + Extract<Self, I2>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, _: &S, dt: Element) {
        transition
            .block::<Translation3<_>, Self, _, _>()
            .fill_diagonal(dt);
    }
}

impl<T1: ErrorState, T2: ErrorState> ErrorState for (T1, T2)
where
    Self: DoF,
{
    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        self.0.inject(inj.rows_generic(0, T1::DoF::name()));
        self.1
            .inject(inj.rows_generic(T1::DoF::DIM, T2::DoF::name()));
    }
}
