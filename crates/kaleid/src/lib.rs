//! A library for building a Extendable Error‑State Kalman Filter.

use core::ops::{Deref, DerefMut, Sub};

use nalgebra::{DefaultAllocator, allocator::Allocator};

pub mod covariance;
mod macros;
pub mod observe;
pub mod predict;
pub mod state;
mod tuples;
pub mod uncertain;

pub use kaleid_macros::*;
use num_traits::{One, Zero};
use simba::scalar::SupersetOf;
pub use state::KFState;

pub use covariance::Cov;
use uncertain::Uncertained;

pub struct Eskf<S: KFState>
where
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub uncertainty: Uncertained<S>,
    pub process_cov: Cov<S>,
    last_update_time: KFTime<S::Element>,
}

#[derive(Debug, Default, Clone)]
struct KFTime<T> {
    pub predict: T,
    pub observe: T,
}

impl<S> Eskf<S>
where
    S: KFState<Element: One + Zero + SupersetOf<f64>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub fn new(process_cov: Cov<S>, timestamp_init: S::Element) -> Self
    where
        S: Default,
    {
        let state = S::default();
        Self::new_with_state(state, process_cov, timestamp_init)
    }

    pub fn new_with_state(state: S, process_cov: Cov<S>, timestamp_init: S::Element) -> Self {
        let uncertainty = Uncertained::new(state);
        Self {
            uncertainty,
            process_cov,
            last_update_time: KFTime {
                predict: timestamp_init.clone(),
                observe: timestamp_init,
            },
        }
    }
}

impl<S: KFState> Deref for Eskf<S>
where
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    type Target = Uncertained<S>;

    fn deref(&self) -> &Self::Target {
        &self.uncertainty
    }
}

impl<S: KFState> DerefMut for Eskf<S>
where
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uncertainty
    }
}

impl<T> Sub for KFTime<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            predict: self.predict - rhs.predict,
            observe: self.observe - rhs.observe,
        }
    }
}

/// #Example
/// ```
/// const N: usize = 5;
/// let mut sum = 0;
/// while let mut i = 0
///     && i < N
///     && let i = crate::incr(&mut i)
/// {
///     sum += i;
/// }
/// assert_eq!(sum, 10);
/// ```
const fn incr(i: &mut usize) -> usize {
    std::mem::replace(i, *i + 1)
}

/// Num increasing until `end`, can be used to create a while let loop
///
/// #Example
/// ```
/// const N: usize = 5;
/// let mut sum = 0;
/// while let mut i = 0
///     && let Some(i) = crate::incr_until(&mut i, N)
/// {
///     sum += i;
/// }
/// assert_eq!(sum, 10);
/// ```
const fn incr_until(i: &mut usize, end: usize) -> Option<usize> {
    let old = incr(i);
    if old < end { Some(old) } else { None }
}
