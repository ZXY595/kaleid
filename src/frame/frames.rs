use nalgebra::{IsometryMatrix3, Matrix3, Point3, Scalar, U3};

use crate::eskf::state::KFState;

use super::Framed;

#[derive(Debug)]
pub struct BodyFrame;

#[derive(Debug)]
pub struct ImuFrame;

#[derive(Debug)]
pub struct WorldFrame;

pub type IsometryFramed<T, F> = Framed<IsometryMatrix3<T>, F>;
pub type CrossMatrixFramed<T, F> = Framed<Matrix3<T>, F>;

pub type FramedPoint<T, F> = Framed<Point3<T>, F>;
pub type BodyPoint<T> = Framed<Point3<T>, BodyFrame>;
pub type ImuPoint<T> = Framed<Point3<T>, ImuFrame>;
pub type WorldPoint<T> = Framed<Point3<T>, WorldFrame>;

impl<T: Scalar, F> KFState for FramedPoint<T, F> {
    type Element = T;
    type Dim = U3;
}

impl<T: Scalar, F> KFState for &Framed<Point3<T>, F> {
    type Element = T;
    type Dim = U3;
}
