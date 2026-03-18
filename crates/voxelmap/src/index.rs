use std::{
    hash::{Hash, Hasher},
    ops::Deref,
};

use derive_deref::{Deref, DerefMut};
use nalgebra::Point3;

use crate::Element;

#[derive(Debug, Deref, DerefMut)]
pub struct VoxelIndex(Point3<i64>);

impl From<Point3<Element>> for VoxelIndex {
    fn from(value: Point3<Element>) -> Self {
        Self(value.map(|p| p as i64))
    }
}

impl VoxelIndex {
    #[inline]
    pub fn new(point: &Point3<Element>, voxel_size: Element) -> Self {
        point.map(|p| p / voxel_size).into()
    }
}

impl Hash for VoxelIndex {
    /// see also Optimized Spatial Hashing for Collision Detection of Deformable Objects,
    /// Matthias Teschner et. al., VMV 2003
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        hasher.write_i64(((self.x * 73856093) ^ (self.y * 471943) ^ (self.z * 83492791)) % 10000000)
    }
}

/// The [`Hash`] implementation of [`WorldPoint<i64>`] invokes [`write_i64`](Hasher::write_i64)
/// method exactly once.
impl nohash_hasher::IsEnabled for VoxelIndex {}

impl PartialEq for VoxelIndex {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other)
    }
}

impl Eq for VoxelIndex {}
