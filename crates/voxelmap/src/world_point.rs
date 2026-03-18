use std::ops::Deref;

use nalgebra::{IsometryMatrix3, Matrix, Matrix3, Point3, Storage, U3};

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

impl UncertainWorldPoint {
    pub fn new(
        body_point: &UncertainBodyPoint,
        imu_to_world: &IsometryMatrix3<Element>,
        body_to_world: &IsometryMatrix3<Element>,
        cross_matrix_imu: &Matrix3<Element>,
        rot_cov: Matrix<Element, U3, U3, impl Storage<Element, U3, U3>>,
        trans_cov: Matrix<Element, U3, U3, impl Storage<Element, U3, U3>>,
    ) -> Self
where {
        let point = body_to_world * body_point.point;

        let mut cov = trans_cov.into_owned();

        cov.quadform_tr(
            1.0,
            body_to_world.rotation.matrix(),
            &body_point.covariance,
            1.0,
        );
        cov.quadform_tr(
            1.0,
            &(imu_to_world.rotation * cross_matrix_imu),
            &rot_cov,
            1.0,
        );

        Self {
            point,
            covariance: cov,
        }
    }
}
