use std::any::TypeId;

use crate::Element;

use derive_deref::{Deref, DerefMut};
pub use nalgebra::Const;
use nalgebra::{
    DimName, Rotation3, SVector, Storage, Translation3, UnitQuaternion, Vector, Vector3,
};

use crate::predict::{BuildTransition, TransitionViewMut};

pub struct Members(&'static [(TypeId, usize)]);

impl Members {
    pub const fn new(items: &'static [(TypeId, usize)]) -> Self {
        Self(items)
    }

    pub const fn single<T: ErrorState>() -> (TypeId, usize) {
        (TypeId::of::<T>(), T::DoF::DIM)
    }

    pub fn position(&self, target: TypeId) -> Option<usize> {
        self.0.iter().position(|&(id, _)| id == target)
    }

    pub fn offset(&self, target: TypeId) -> Option<usize> {
        let pos = self.position(target)?;
        self.0
            .iter()
            .take(pos)
            .map(|&(_, dof)| dof)
            .sum::<usize>()
            .into()
    }
}

pub trait ErrorState: 'static {
    type DoF: DimName;
    const MEMBERS: Members = Members(&[]);

    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>);

    #[inline]
    fn offset<T: 'static>() -> usize
    where
        Self: AsRef<T>,
    {
        let offset = Self::MEMBERS.offset(TypeId::of::<T>());
        // # Safety:
        //
        // `Self` implementes `AsRef<T>`, so `offset` is always `Some`.
        unsafe { offset.unwrap_unchecked() }
    }

    #[inline]
    fn offset_of<S: ErrorState>() -> Result<usize, String> {
        let pos = Self::MEMBERS
            .0
            .windows(S::MEMBERS.0.len())
            .position(|w| w == S::MEMBERS.0)
            .ok_or_else(|| {
                format!(
                    "{} is not found in {}",
                    std::any::type_name::<S>(),
                    std::any::type_name::<Self>()
                )
            })?;
        Ok(Self::MEMBERS
            .0
            .iter()
            .take(pos)
            .map(|&(_, dof)| dof)
            .sum::<usize>())
    }
}

pub use kaleid_macros::ErrorState;

impl<const D: usize> ErrorState for SVector<Element, D> {
    type DoF = Const<D>;

    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self += inj
    }
}

impl ErrorState for Translation3<Element> {
    type DoF = Const<3>;

    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self *= Translation3::from(inj.into_owned())
    }
}

impl ErrorState for UnitQuaternion<Element> {
    type DoF = Const<3>;

    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self *= UnitQuaternion::from_scaled_axis(inj.into_owned())
    }
}

impl ErrorState for Rotation3<Element> {
    type DoF = Const<3>;

    fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
        *self *= Rotation3::from_scaled_axis(inj.into_owned())
    }
}
#[derive(Debug, ErrorState, Deref, DerefMut)]
pub struct Velocity(#[DoF = 3] Vector3<Element>);

impl<S: ErrorState> BuildTransition<S> for Velocity
where
    S: AsRef<Translation3<Element>> + AsRef<Self>,
{
    fn build_transition(transition: &mut TransitionViewMut<S>, _: &S, dt: Element) {
        transition
            .block::<Translation3<_>, Self>()
            .fill_diagonal(dt);
    }
}
