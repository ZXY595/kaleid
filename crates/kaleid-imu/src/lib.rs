use kaleid::{KFState, bundle};
use nalgebra::{U3, Vector3};

#[bundle(export)]
pub struct ImuState<T>(
    crate::Velocity<T>,
    crate::Accel<T>,
    crate::Gyro<T>,
    crate::Gravity<T>,
);

#[test]
fn test_from_bundle() {
    let mut s = ImuState::<f32>(
        Velocity(Vector3::zeros()),
        Accel(Vector3::zeros()),
        Gyro(Vector3::zeros()),
        Gravity(Vector3::zeros()),
    );
    use kaleid::state::bundle::FromBundle;
    let _ = <(&mut Velocity<f32>, &mut Accel<f32>)>::from_bundle(&mut s);
}

#[derive(Debug)]
pub struct Velocity<T>(pub Vector3<T>);

impl<T> KFState for Velocity<T> {
    type Element = T;
    type Dim = U3;
}

/// Linear acceleration in the imu frame
#[derive(Debug, Default)]
pub struct Accel<T>(pub Vector3<T>);

impl<T> KFState for Accel<T> {
    type Element = T;
    type Dim = U3;
}

/// Angular velocity in the imu frame
#[derive(Debug, Default)]
pub struct Gyro<T>(pub Vector3<T>);

impl<T> KFState for Gyro<T> {
    type Element = T;
    type Dim = U3;
}

#[derive(Debug)]
pub struct Gravity<T>(pub Vector3<T>);

impl<T> KFState for Gravity<T> {
    type Element = T;
    type Dim = U3;
}
