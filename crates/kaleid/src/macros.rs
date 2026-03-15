/// A macro to generate a [`nalgebra::Matrix`] type that storage type is `impl` [`nalgebra::Storage`].
///
/// # Example
/// ```rust
/// use nalgebra::{U3, U4};
/// use odometries::ImplMatrix;
///
/// fn test(h: ImplMatrix!(f32, U3, U4)) {
///     let _ = h.transpose();
/// }
/// ```
#[macro_export]
macro_rules! ImplMatrix {
    ( $name:ty, $rows:ty, $cols:ty, $storage:ident ) => {
        nalgebra::Matrix<$name, $rows, $cols, impl nalgebra::$storage<$name, $rows, $cols>>
    };
    ( $name:ty, $rows:ty, $cols:ty ) => {
        $crate::ImplMatrix!($name, $rows, $cols, Storage)
    };
    ( $name:ty, $dim:ty ) => {
        $crate::ImplMatrix!($name, $dim, $dim)
    };
}

#[macro_export]
macro_rules! ImplVector {
    ( $name:ty, $rows:ty, $storage:ident ) => {
        nalgebra::Vector<$name, $rows, impl nalgebra::$storage<$name, $rows, nalgebra::U1>>
    };
    ( $name:ty, $rows:ty ) => {
        $crate::ImplVector!($name, $rows, Storage)
    };
}
