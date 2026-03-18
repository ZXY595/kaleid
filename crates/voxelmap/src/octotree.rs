mod branch;
mod leaf;

use branch::{Branch, BranchIndex};
use leaf::Leaf;

use std::num::NonZero;

use extension_traits::extension;
use nalgebra::Point3;
use slab::Slab;
use smallvec::SmallVec;

use crate::{
    Element,
    index::VoxelIndex,
    plane::{PlaneConfig, UncertainPlane},
    world_point::UncertainWorldPoint,
};

pub struct Root {
    storage: Slab<Node>,
    /// The key of the root node in the `storage`.
    root_key: usize,
}

struct Node {
    kind: NodeKind,
    state: NodeState,
}

struct NodeState {
    center: Point3<Element>,
    /// The quarter length of the side of the node.
    quarter_side_length: Element,
    /// Current depth of the node in the tree.
    depth: u8,
}

#[expect(clippy::large_enum_variant)]
enum NodeKind {
    Branch(Branch),
    Leaf(Leaf),
}

impl Root {
    pub fn new(point: &Point3<Element>, voxel_size: Element) -> Self {
        let center = point.map(|x| x.floor() + voxel_size / 2.0);
        let root = Node::new(
            NodeKind::Leaf(Leaf::default()),
            NodeState {
                center,
                quarter_side_length: voxel_size / 4.0,
                depth: 0,
            },
        );
        let mut slab = Slab::with_capacity(1);
        Self {
            root_key: {
                let root_key = slab.insert(root);
                debug_assert_eq!(root_key, 0);
                root_key
            },
            storage: slab,
        }
    }

    pub fn insert(&mut self, point: UncertainWorldPoint, config: &impl PlaneConfig) {
        RecursiveInsertIter {
            current: self.root_key,
            points: SmallVec::from_buf([point]).into_iter(),
            config,
            storage: &mut self.storage,
        }
        .for_each(drop);
    }

    pub fn iter_planes(&self) -> impl Iterator<Item = &UncertainPlane> {
        self.storage
            .iter()
            .filter_map(|(_, node)| match &node.kind {
                NodeKind::Leaf(leaf) => leaf.plane.as_ref(),
                _ => None,
            })
    }

    pub fn nearest_voxel(
        &self,
        world_point: &Point3<Element>,
        mut index: VoxelIndex,
    ) -> VoxelIndex {
        let NodeState {
            center,
            quarter_side_length,
            ..
        } = &self.storage[self.root_key].state;

        world_point
            .iter()
            .zip(center.iter())
            .zip(index.iter_mut())
            .for_each(|((point, center), coord)| {
                if *point > center + quarter_side_length {
                    *coord += 1;
                } else if *point < center - quarter_side_length {
                    *coord -= 1;
                }
            });
        index
    }
}

impl Node {
    fn new(kind: NodeKind, state: NodeState) -> Self {
        Self { kind, state }
    }

    fn new_child(state: &NodeState, index: &BranchIndex, point: UncertainWorldPoint) -> Self {
        Self::new(NodeKind::Leaf(Leaf::from_point(point)), state.deeper(index))
    }
}

impl NodeState {
    fn deeper(&self, index: &BranchIndex) -> Self {
        let child_center = self.center
            + index.map(|x| {
                if x {
                    self.quarter_side_length
                } else {
                    -self.quarter_side_length
                }
            });
        Self {
            center: child_center,
            quarter_side_length: self.quarter_side_length / 2.0,
            depth: self.depth + 1,
        }
    }
}

struct RecursiveInsertIter<'config, 'store, C: PlaneConfig> {
    current: usize,
    /// The points to be inserted.
    points: smallvec::IntoIter<UncertainWorldPoint, 1>,
    config: &'config C,
    storage: &'store mut Slab<Node>,
}

impl<C: PlaneConfig> Iterator for RecursiveInsertIter<'_, '_, C> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.points.next()?;

        let (last_node, index) = PathIter {
            current: self.current,
            point: &point,
            storage: self.storage,
        }
        .last()
        .unzip();

        let current_key = last_node.de_niche();
        let vacant_key = self.storage.vacant_key();
        let node = &mut self.storage[current_key];
        match (&mut node.kind, index) {
            (NodeKind::Branch(branch), Some(index)) => {
                branch[&index] = NonZero::new(vacant_key);
                let new_leaf = Node::new_child(&node.state, &index, point);
                debug_assert_eq!(self.storage.insert(new_leaf), vacant_key);
            }
            (NodeKind::Leaf(leaf), _) => {
                if let Err(points_not_plane) = leaf
                    .insert(point, self.config, node.state.depth)
                    .map_err(SmallVec::from_vec)
                {
                    // pruning the leaf and creating a new branch
                    node.kind = NodeKind::Branch(Default::default());
                    self.points = points_not_plane.into_iter();
                    self.current = current_key;
                };
            }
            _ => {}
        };
        Some(())
    }
}

struct PathIter<'p, 'store> {
    /// [`None`] when `current` is [`Root`]
    current: usize,
    point: &'p UncertainWorldPoint,
    storage: &'store Slab<Node>,
}

impl<'p, 'store> Iterator for PathIter<'p, 'store> {
    type Item = (NonZero<usize>, BranchIndex);

    fn next(&mut self) -> Option<Self::Item> {
        let current_node = &self.storage[self.current];

        match &current_node.kind {
            NodeKind::Branch(branch) => {
                let index = BranchIndex::new(self.point, &current_node.state.center);
                let child = branch[&index]?;
                self.current = child.get();
                Some((child, index))
            }
            NodeKind::Leaf(_) => None,
        }
    }
}

#[extension(trait DeNiche)]
impl Option<NonZero<usize>> {
    fn de_niche(self) -> usize {
        self.map_or(0, NonZero::get)
    }
}
