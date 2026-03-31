pub mod body_point;
mod index;
mod octotree;
pub mod plane;
pub mod residual;
pub mod world_point;

use nohash_hasher::IntMap;

use crate::{
    index::VoxelIndex,
    plane::{PlaneConfig, UncertainPlane},
    world_point::UncertainWorldPoint,
};

pub use octotree::Root as OctoTreeRoot;

pub type Element = f64;

pub struct VoxelMap<C: Config> {
    roots: IntMap<VoxelIndex, OctoTreeRoot>,
    config: C,
}

pub trait Config: PlaneConfig {
    /// residual sigma factor, larger value means more uncertain
    fn sigma_ratio(&self) -> Element {
        3.0
    }

    /// voxel size in the voxel grid
    fn voxel_size(&self) -> Element {
        0.5
    }

    /// map size for map sliding window
    fn map_size(&self) -> usize {
        200
    }

    /// delta pose change threshold to update map sliding window
    fn sliding_thresh(&self) -> Element {
        8.0
    }
}

impl<C: Config> VoxelMap<C> {
    pub fn new(config: C) -> Self {
        Self {
            roots: IntMap::default(),
            config,
        }
    }

    pub fn iter_planes(&self) -> impl Iterator<Item = &UncertainPlane> {
        self.roots.values().flat_map(octotree::Root::iter_planes)
    }
}

impl<C: Config> VoxelMap<C> {
    pub fn insert(&mut self, point: UncertainWorldPoint) {
        let voxel_size = self.config.voxel_size();
        let index = VoxelIndex::new(&point, voxel_size);
        self.roots
            .entry(index)
            .or_insert_with(|| octotree::Root::new(&point, voxel_size))
            .insert(point, &self.config);
    }
}

impl<C: Config> Extend<UncertainWorldPoint> for VoxelMap<C> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = UncertainWorldPoint>,
    {
        iter.into_iter().for_each(|point| self.insert(point));
    }
}
