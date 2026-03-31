use std::ops::Deref;

use derive_deref::{Deref, DerefMut};
use nalgebra::{IsometryMatrix3, Matrix, Matrix3, Point3, Rotation3, Storage, U3};

use crate::{Element, body_point::UncertainBodyPoint};

pub struct UncertainWorldPoint {
    pub point: Point3<Element>,
    pub covariance: Matrix3<Element>,
}

impl Deref for UncertainWorldPoint {
    type Target = Point3<Element>;

    fn deref(&self) -> &Self::Target {
        &self.point
    }
}

#[derive(Deref, DerefMut)]
pub struct RotJocobian(Matrix3<Element>);

impl RotJocobian {
    pub fn new(imu_to_world: &Rotation3<Element>, imu_point: &Point3<Element>) -> Self {
        Self(imu_to_world * imu_point.coords.cross_matrix())
    }
}

impl UncertainWorldPoint {
    pub fn new(
        body_point: &UncertainBodyPoint,
        body_to_world: &IsometryMatrix3<Element>,
        rot_jocobian: &RotJocobian,
        trans_cov: Matrix<Element, U3, U3, impl Storage<Element, U3, U3>>,
        rot_cov: Matrix<Element, U3, U3, impl Storage<Element, U3, U3>>,
    ) -> Self
where {
        let world_point = body_to_world * body_point.point;

        let mut cov = trans_cov.into_owned();
        cov.quadform_tr(1.0, rot_jocobian, &rot_cov, 1.0);
        cov.quadform_tr(
            1.0,
            body_to_world.rotation.matrix(),
            &body_point.covariance,
            1.0,
        );

        Self {
            point: world_point,
            covariance: cov,
        }
    }

    pub fn new_no_state(
        body_point: &UncertainBodyPoint,
        body_to_world: &IsometryMatrix3<Element>,
    ) -> Self {
        let world_point = body_to_world * body_point.point;

        let mut cov = Matrix3::zeros();
        cov.quadform_tr(
            1.0,
            body_to_world.rotation.matrix(),
            &body_point.covariance,
            0.0,
        );

        Self {
            point: world_point,
            covariance: cov,
        }
    }
}
