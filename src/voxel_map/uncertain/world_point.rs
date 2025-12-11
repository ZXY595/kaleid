use std::ops::Deref;

use nalgebra::{DefaultAllocator, Matrix3, RealField, Scalar, U3, allocator::Allocator};

use crate::{
    eskf::{
        Covariance,
        state::{
            KFState, SubStateOf,
            common::{PositionState, RotationState},
        },
    },
    frame::{
        Framed, IsometryFramed,
        frames::{BodyFrame, ImuFrame, WorldFrame},
    },
    voxel_map::uncertain::UncertainBodyPoint,
};

use super::UncertainWorldPoint;

pub struct UncertainPoint<T: Scalar> {
    pub world_point: UncertainWorldPoint<T>,
    pub cross_matrix_imu: Matrix3<T>,
}

impl<T> UncertainBodyPoint<T>
where
    T: RealField,
{
    pub fn to_uncertain_world_point<S>(
        self,
        imu_to_world: &IsometryFramed<T, fn(ImuFrame) -> WorldFrame>,
        body_to_world: &IsometryFramed<T, fn(BodyFrame) -> WorldFrame>,
        cross_matrix_imu: Framed<&Matrix3<T>, ImuFrame>,
        eskf_cov: &Covariance<S>,
    ) -> UncertainWorldPoint<T>
    where
        S: KFState<Element = T>,
        RotationState<T>: SubStateOf<S, Dim = U3>,
        PositionState<T>: SubStateOf<S, Dim = U3>,
        DefaultAllocator: Allocator<S::Dim, S::Dim>,
    {
        let world_point = self.deref() * body_to_world;
        let rot_cov = eskf_cov.sub_covariance::<RotationState<T>>();
        let pos_cov = eskf_cov.sub_covariance::<PositionState<T>>();

        let mut cov = pos_cov.into_owned();
        cov.quadform_tr(
            T::one(),
            body_to_world.rotation.matrix(),
            &self.cov,
            T::one(),
        );
        cov.quadform_tr(
            T::one(),
            &(&imu_to_world.rotation * *cross_matrix_imu),
            &rot_cov,
            T::one(),
        );

        UncertainWorldPoint::new_with_cov(world_point, cov)
    }
}
