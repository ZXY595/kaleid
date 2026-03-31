use std::{cmp::Ordering, ops::Deref};

use nalgebra::{IsometryMatrix3, Vector3};

use crate::{
    Config, Element, UncertainWorldPoint, VoxelMap, body_point::UncertainBodyPoint,
    index::VoxelIndex, octotree, plane::UncertainPlane,
};

pub struct Residual<'a> {
    pub plane: &'a UncertainPlane,
    pub distance_to_plane: Element,
    distance_variance: Element,
    distance_to_plane_squared: Element,
    distance_variance_sqrt: Element,
}

pub struct InvalidResidual<'a> {
    /// the root of the oct tree where the residual of the given point was not found
    voxel_root: &'a octotree::Root,
    voxel_index: VoxelIndex,
}

impl<C> VoxelMap<C>
where
    C: Config,
{
    pub fn get_residual(
        &self,
        point: &UncertainWorldPoint,
    ) -> Result<Residual<'_>, Option<InvalidResidual<'_>>> {
        let index = VoxelIndex::new(point, self.config.voxel_size());
        self.get_residual_by_index(index, point)
    }

    pub fn get_or_nearest_residual(&self, point: &UncertainWorldPoint) -> Option<Residual<'_>> {
        self.get_residual(point)
            .or_else(|err| {
                let InvalidResidual {
                    voxel_root,
                    voxel_index,
                } = err.ok_or(())?;

                let nearest_index = voxel_root.nearest_voxel(point, voxel_index);

                self.get_residual_by_index(nearest_index, point)
                    .map_err(drop)
            })
            .ok()
    }

    fn get_residual_by_index(
        &self,
        index: VoxelIndex,
        point: &UncertainWorldPoint,
    ) -> Result<Residual<'_>, Option<InvalidResidual<'_>>> {
        let voxel_root = self.roots.get(&index).ok_or(None)?;
        let radius_factor = 3.0;

        // TODO: could be optimized by using `rayon`?
        voxel_root
            .iter_planes()
            .map(|plane| {
                let normal = &plane.normal;
                let distance_to_plane =
                    normal.dot(&point.coords) - normal.dot(&plane.center.coords);
                (plane, distance_to_plane, distance_to_plane.powi(2))
            })
            .filter(|(plane, _, distance_to_plane_squared)| {
                let distance_to_center = point.deref() - plane.center;
                let range_distance =
                    (distance_to_center.norm_squared() - distance_to_plane_squared).sqrt();
                range_distance <= radius_factor * plane.radius
            })
            .map(|(plane, distance_to_plane, distance_to_plane_squared)| {
                let distance_variance = plane.distance_variance(point).to_scalar();
                Residual {
                    plane,
                    distance_to_plane,
                    distance_to_plane_squared,
                    distance_variance_sqrt: distance_variance.sqrt(),
                    distance_variance,
                }
            })
            .filter(|residual| {
                residual.distance_to_plane.abs()
                    < self.config.sigma_ratio() * residual.distance_variance_sqrt
            })
            .max_by(|a, b| {
                a.probability()
                    .partial_cmp(&b.probability())
                    // TODO: this might be bad
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or(Some(InvalidResidual {
                voxel_root,
                voxel_index: index,
            }))
    }
}

impl<'a> Residual<'a> {
    #[inline]
    pub fn plane_normal(&self) -> &Vector3<Element> {
        &self.plane.normal
    }

    fn probability(&self) -> Element {
        self.distance_variance_sqrt.recip()
            * (self.distance_to_plane_squared * -0.5 / self.distance_variance).exp()
    }

    pub fn noise(
        &self,
        body_point: &UncertainBodyPoint,
        body_to_world: &IsometryMatrix3<Element>,
    ) -> Element {
        self.plane
            .distance_variance(&UncertainWorldPoint::new_no_state(
                body_point,
                body_to_world,
            ))
            .to_scalar()
    }
}
