use crate::algorithm::lio;
use kaleid::state::{KFState, StateBegin, StateEnd, StateOffset, common::Vector3State};
use nalgebra::{RealField, Scalar};
use odometries_macros::KFState;

#[derive(KFState)]
#[element(T: RealField)]
pub struct State<T>
where
    T: Scalar,
{
    state: lio::state::State<T>,
    kinematic_velocity_bias: KinVelocityBiasState<T>,
    contact_foot_pos: ContactFootPosState<T>,
}

pub struct KinVelocityBias;
type KinVelocityBiasState<T> = Vector3State<T, KinVelocityBias>;

pub struct ContactFootPos;
type ContactFootPosState<T> = Vector3State<T, ContactFootPos>;
