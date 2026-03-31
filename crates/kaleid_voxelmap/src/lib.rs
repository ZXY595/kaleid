#![expect(clippy::toplevel_ref_arg)]

use extension_traits::extension;
use kaleid::{Covariance, Element, error_state::ErrorState};
use nalgebra::{
    DVector, DefaultAllocator, Dyn, IsometryMatrix3, Matrix6xX, Point3, Rotation3, Translation3,
    allocator::Allocator,
};
use voxelmap::{
    VoxelMap,
    body_point::{self, UncertainBodyPoint},
    world_point::{RotJocobian, UncertainWorldPoint},
};

#[derive(ErrorState)]
struct PoseState {
    trans: Translation3<Element>,
    rot: Rotation3<Element>,
}

#[extension(pub trait ObservePoints)]
impl<S: ErrorState> Covariance<S>
where
    DefaultAllocator: Allocator<S::DoF, S::DoF> + Allocator<S::DoF>,
{
    fn observe_points(
        &mut self,
        states: &mut S,
        voxel_map: &VoxelMap<impl voxelmap::Config>,
        body_process_cov: &body_point::ProcessCov,
        body_to_imu: &IsometryMatrix3<Element>,
        body_points: impl IntoIterator<Item = Point3<Element>>,
    ) where
        S: AsRef<Translation3<Element>> + AsRef<Rotation3<Element>>,
    {
        let trans = states.as_ref();
        let rot = states.as_ref();
        let imu_to_world = IsometryMatrix3::from_parts(*trans, *rot);
        let body_to_world = imu_to_world * body_to_imu;

        let (measurement, model, noise) = body_points
            .into_iter()
            .map(|body_point| UncertainBodyPoint::new(body_point, body_process_cov))
            .filter_map(|body_point| {
                let rot_jocobian =
                    RotJocobian::new(&imu_to_world.rotation, &(body_to_imu * body_point.point));

                let world_point = UncertainWorldPoint::new(
                    &body_point,
                    &body_to_world,
                    &rot_jocobian,
                    self.block::<Translation3<Element>, Translation3<Element>>(),
                    self.block::<Rotation3<Element>, Rotation3<Element>>(),
                );
                let residual = voxel_map.get_or_nearest_residual(&world_point)?;

                let plane_normal = residual.plane.normal;
                Some((
                    -residual.distance_to_plane,
                    nalgebra::stack![rot_jocobian.transpose() * plane_normal; plane_normal],
                    10.0 * residual.noise(&body_point, &body_to_world),
                ))
            })
            .collect::<(DVector<Element>, Matrix6xX<Element>, DVector<Element>)>();

        let cross_cov_tr = model.transpose() * self.rows::<PoseState>();

        let innovation_cov = model.transpose() * self.block::<PoseState, PoseState>() * model;

        self.observe::<Dyn>(states, &measurement, &noise, &cross_cov_tr, innovation_cov)
    }
}
