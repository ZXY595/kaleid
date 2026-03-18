use std::{
    num::NonZero,
    ops::{Index, IndexMut},
};

use derive_deref::Deref;
use nalgebra::{Point3, Vector3};

use crate::Element;

#[derive(Default)]
pub struct Branch([[[Option<NonZero<usize>>; 2]; 2]; 2]);

#[derive(Deref)]
pub struct BranchIndex(Vector3<bool>);

impl BranchIndex {
    pub fn new(point: &Point3<Element>, node_center: &Point3<Element>) -> Self {
        Self((point - node_center).map(|x| x > 0.0))
    }
}

impl Index<&BranchIndex> for Branch {
    type Output = Option<NonZero<usize>>;

    fn index(&self, BranchIndex(index): &BranchIndex) -> &Self::Output {
        &self.0[index.x as usize][index.y as usize][index.z as usize]
    }
}

impl IndexMut<&BranchIndex> for Branch {
    fn index_mut(&mut self, BranchIndex(index): &BranchIndex) -> &mut Self::Output {
        &mut self.0[index.x as usize][index.y as usize][index.z as usize]
    }
}
