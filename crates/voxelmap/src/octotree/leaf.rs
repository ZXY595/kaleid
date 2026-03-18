use crate::{PlaneConfig, UncertainPlane, UncertainWorldPoint, plane::PlaneInitError};

pub struct Leaf {
    pub(super) plane: Option<UncertainPlane>,
    /// Cached points for initializing a new plane or updating the existing plane.
    /// If the cached points is `None`, the plane is not need to be updated.
    cached_points: Option<Vec<UncertainWorldPoint>>,
}

impl Leaf {
    pub fn from_point(p: UncertainWorldPoint) -> Self {
        Self {
            cached_points: Some(vec![p]),
            ..Default::default()
        }
    }

    /// ## Returns
    ///
    /// - `Err(_)` if the try to create a plane but not vilid, this might need a node pruning.
    pub fn insert(
        &mut self,
        point: UncertainWorldPoint,
        config: &impl PlaneConfig,
        depth: u8,
    ) -> Result<(), Vec<UncertainWorldPoint>> {
        let Leaf {
            plane,
            cached_points,
        } = self;

        let Some(points) = cached_points else {
            // cached points is disabled, the plane is not need to be updated.
            return Ok(());
        };

        points.push(point);

        let points_len = points.len();

        // drop the old plane if it needs update
        let _ = plane.take_if(|_| config.need_update(points_len));

        if plane.is_none() {
            *plane = UncertainPlane::new(points, config)
                .map(Some)
                .or_else(|err| match err {
                    PlaneInitError::EigenValueTooBig { .. } if config.leaf_prunable(depth) => {
                        Err(std::mem::take(points))
                    }
                    _ => Ok(None),
                })?;
        }

        if config.is_plane_max(points_len) {
            // disable plane initialize and update
            *cached_points = None;
        }
        Ok(())
    }
}

impl Default for Leaf {
    fn default() -> Self {
        Self {
            plane: None,
            cached_points: Some(Vec::new()),
        }
    }
}
