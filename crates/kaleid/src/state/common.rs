use crate::ImplVector;

use super::KFState;

use nalgebra::{IsometryMatrix3, RealField, Rotation3, Translation3, U6};

#[derive(Debug)]
pub struct Pose<T>(pub IsometryMatrix3<T>);

impl<T: RealField> KFState for Pose<T> {
    type Element = T;
    type Dim = U6;

    fn gain(&mut self, rhs: ImplVector!(Self::Element, Self::Dim)) {
        self.0 *= Rotation3::new(rhs.fixed_rows(0));
        self.0 *= Translation3::from(rhs.fixed_rows(3).into_owned());
    }
}

// #[derive(Debug)]
// pub struct Rotation<T>(pub Vector3<T>);
//
// impl<T: Scalar> KFState for Rotation<T> {
//     type Element = T;
//     type Dim = U0;
// }
//
// #[derive(Debug)]
// pub struct Position<T>(pub Vector3<T>);
//
// impl<T: Scalar> KFState for Position<T> {
//     type Element = T;
//     type Dim = U0;
// }
