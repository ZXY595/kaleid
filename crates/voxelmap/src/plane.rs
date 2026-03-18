use std::{
    iter::Sum,
    ops::{Deref, Not},
};

use nalgebra::{Matrix1, Matrix3, Matrix6, Point3, RowVector3, SymmetricEigen, Vector3, stack};

use crate::{Element, world_point::UncertainWorldPoint};

pub struct Plane {
    pub normal: Vector3<Element>,
    pub center: Point3<Element>,
    pub radius: Element,
}

pub struct UncertainPlane {
    plane: Plane,
    covariance: Matrix6<Element>,
}

impl Deref for UncertainPlane {
    type Target = Plane;

    fn deref(&self) -> &Self::Target {
        &self.plane
    }
}

pub trait PlaneConfig {
    fn leaf_prunable(&self, depth: u8) -> bool {
        depth < 4
    }

    /// minimum number of points to init a tree
    fn plane_initializable(&self, points_len: usize) -> bool {
        points_len >= 5
    }

    /// minimum number of points to update a plane
    fn need_update(&self, points_len: usize) -> bool {
        points_len.is_multiple_of(5)
    }

    /// maximum eigen value of a plane to be considered as a valid plane
    fn plane_eigen_is_valid(&self, eigen: Element) -> bool {
        eigen >= 0.1
    }

    /// maximum number of points for a tree
    fn is_plane_max(&self, points_len: usize) -> bool {
        points_len >= 50
    }
}

pub enum PlaneInitError {
    TooFewPoints,
    EigenValueTooBig { eigen: Element },
}

impl UncertainPlane {
    pub fn new(
        points: &[UncertainWorldPoint],
        config: &impl PlaneConfig,
    ) -> Result<Self, PlaneInitError> {
        if config.plane_initializable(points.len()).not() {
            return Err(PlaneInitError::TooFewPoints);
        }

        let sum = points
            .iter()
            .map(|uncertain_point| &uncertain_point.coords)
            .sum::<VectorSquareSum>();

        let points_count = sum.count();
        let (center, covariance) = sum.mean();

        let SymmetricEigen {
            eigenvectors,
            eigenvalues,
        } = covariance.symmetric_eigen();

        let (min_eigen_index, min_eigen_value) = eigenvalues.argmin();

        if config.plane_eigen_is_valid(min_eigen_value).not() {
            return Err(PlaneInitError::EigenValueTooBig {
                eigen: min_eigen_value,
            });
        }

        let min_eigenvector = eigenvectors.column(min_eigen_index);

        let points_count = points_count as f64;

        let covariance = points
            .iter()
            .map(|point| {
                let rows = eigenvalues
                    .iter()
                    .zip(eigenvectors.column_iter())
                    .map(|(eigenvalue, eigenvector)| {
                        let eigen_diff = min_eigen_value - eigenvalue;
                        if eigen_diff == 0.0 {
                            return RowVector3::zeros();
                        }

                        (point.deref() - center).coords.transpose() / (points_count * eigen_diff)
                            * (eigenvector * min_eigenvector.transpose()
                                + min_eigenvector * eigenvector.transpose())
                    })
                    .flat_map(|row| row.data.0.into_iter())
                    .map(|[x]| x);

                let normal_error = eigenvectors * Matrix3::from_row_iterator(rows);
                let position_error = Matrix3::from_diagonal_element(points_count.recip());

                #[expect(clippy::toplevel_ref_arg)]
                let error_matrix = stack![normal_error; position_error];

                // TODO: can we avoid init with zeros?
                let mut cov = Matrix6::zeros();
                cov.quadform_tr(1.0, &error_matrix, &point.covariance, 0.0);
                cov
            })
            .sum::<Matrix6<Element>>();

        let normal = eigenvectors.column(min_eigen_index).into();

        Ok(Self {
            plane: Plane {
                normal,
                center: center.into(),
                radius: eigenvalues.max().sqrt(),
            },
            covariance,
        })
    }

    pub fn sigma_to(&self, world_point: &UncertainWorldPoint) -> Matrix1<Element> {
        let distance_error = world_point.point - self.plane.center;
        let normal_error = -&self.plane.normal;

        #[expect(clippy::toplevel_ref_arg)]
        let error_matrix = stack![distance_error; normal_error];

        let mut sigma = Matrix1::zeros();
        sigma.quadform(1.0, &self.covariance, &error_matrix, 0.0);
        sigma.quadform(1.0, &world_point.covariance, &self.plane.normal, 1.0);
        sigma
    }
}

struct VectorSquareSum {
    count: usize,
    sum: Vector3<Element>,
    square_sum: Matrix3<Element>,
}

impl VectorSquareSum {
    pub fn mean(&self) -> (Vector3<Element>, Matrix3<Element>) {
        let count = self.count as f64;
        let mean = self.sum / count;
        let covariance = self.square_sum / count - mean * mean.transpose();
        (mean, covariance)
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for VectorSquareSum {
    fn default() -> Self {
        Self {
            count: 0,
            sum: Vector3::zeros(),
            square_sum: Matrix3::zeros(),
        }
    }
}

impl<'a> Sum<&'a Vector3<Element>> for VectorSquareSum {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Vector3<Element>>,
    {
        iter.fold(Self::default(), |mut acc, current| {
            acc.count += 1;
            acc.sum += current;
            acc.square_sum += current * current.transpose();
            acc
        })
    }
}
