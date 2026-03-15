use kaleid::{State, StateWithBias, state::Bias};
use nalgebra::{ClosedAddAssign, RealField, Scalar, Vector3};
use odometries_macros::KFState;

#[derive(KFState)]
#[element(T: RealField)]
pub struct ImuState<T> {
    pub velocity: Velocity<T>,
    pub acc: Accel<T>,
    pub gyro: Gyro<T>,
    pub gravity: Gravity<T>,
}

#[derive(Debug)]
pub struct Velocity<T>(pub Vector3<T>);

/// Linear acceleration in the [`ImuFrame`](crate::frame::ImuFrame).
#[derive(Debug, Default)]
pub struct Accel<T>(pub Vector3<T>);

/// Angular velocity in the [`ImuFrame`](crate::frame::ImuFrame).
#[derive(Debug, Default)]
pub struct Gyro<T>(pub Vector3<T>);

#[derive(Debug)]
pub struct Gravity<T>(pub Vector3<T>);

impl<T: RealField> Default for ImuState<T> {
    fn default() -> Self {
        Self {
            velocity: Default::default(),
            acc: Default::default(),
            gyro: Default::default(),
            gravity: Default::default(),
        }
    }
}

impl<T> Bias for Accel<T>
where
    T: Scalar + ClosedAddAssign,
{
    fn bias(&self, bias: &Self) -> Self {
        Self(self.0.bias(&bias.0))
    }
}

impl<T> Bias for Gyro<T>
where
    T: Scalar + ClosedAddAssign,
{
    fn bias(&self, bias: &Self) -> Self {
        Self(self.0.bias(&bias.0))
    }
}
