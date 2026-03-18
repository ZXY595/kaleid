use std::ops::Deref;

use extension_traits::extension;
use nalgebra::{Matrix1, Matrix2, Matrix3, Matrix3x2, Point3, Vector3};

use crate::Element;

#[derive(Debug)]
pub struct UncertainBodyPoint {
    pub point: Point3<Element>,
    pub covariance: Matrix3<Element>,
}

impl Deref for UncertainBodyPoint {
    type Target = Point3<Element>;

    fn deref(&self) -> &Self::Target {
        &self.point
    }
}

#[derive(Debug, Clone)]
pub struct ProcessCov {
    distance_cov: Matrix1<Element>,
    direction_cov: Matrix2<Element>,
}

impl ProcessCov {
    pub fn new(distance: Element, direction: Element) -> Self {
        let distance_cov = Matrix1::new(distance.powi(2));
        let direction_cov = Matrix2::from_diagonal_element(direction.to_radians().sin().powi(2));

        Self {
            distance_cov,
            direction_cov,
        }
    }
}

impl UncertainBodyPoint {
    pub fn new(point: Point3<Element>, process_cov: ProcessCov) -> Self {
        let distance = point.coords.norm();
        let direction = point.coords.normalize();

        let base1 = Vector3::new(1.0, 1.0, {
            -(direction.x + direction.y) / direction.z.or_substitute()
        })
        .normalize();

        let base2 = base1.cross(&direction).normalize();

        let point_base_coords =
            direction.cross_matrix() * distance * Matrix3x2::from_columns(&[base1, base2]);

        let mut cov = Matrix3::zeros();
        cov.quadform_tr(1.0, &direction, &process_cov.distance_cov, 0.0);
        cov.quadform_tr(1.0, &point_base_coords, &process_cov.direction_cov, 1.0);

        Self {
            point,
            covariance: cov,
        }
    }
}

#[extension(trait Substitute)]
impl Element {
    fn or_substitute(self) -> Element {
        if self == 0.0 { 0.0001 } else { self }
    }
}
