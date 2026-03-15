use nalgebra::{ComplexField, Matrix3, Scalar, Vector3};
use num_traits::{Zero, float::FloatCore};
use std::iter::Sum;

pub trait ToRadians {
    fn to_radians(self) -> Self;
}

impl<T: FloatCore> ToRadians for T {
    #[inline(always)]
    fn to_radians(self) -> Self {
        <T as FloatCore>::to_radians(self)
    }
}

pub struct VectorSquareSum<T: Scalar> {
    count: usize,
    sum: Vector3<T>,
    square_sum: Matrix3<T>,
}

impl<T> VectorSquareSum<T>
where
    T: ComplexField,
{
    pub fn mean(&self) -> (Vector3<T>, Matrix3<T>) {
        let count: T = nalgebra::convert(self.count as f64);
        let mean = &self.sum / count.clone();
        let covariance = &self.square_sum / count - &mean * mean.transpose();
        (mean, covariance)
    }
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<T> Default for VectorSquareSum<T>
where
    T: Scalar + Zero,
{
    fn default() -> Self {
        Self {
            count: 0,
            sum: Vector3::zeros(),
            square_sum: Matrix3::zeros(),
        }
    }
}

impl<'a, T> Sum<&'a Vector3<T>> for VectorSquareSum<T>
where
    T: ComplexField,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Vector3<T>>,
    {
        iter.fold(Self::default(), |mut acc, current| {
            acc.count += 1;
            acc.sum += current;
            acc.square_sum += current * current.transpose();
            acc
        })
    }
}

pub trait CollectTo: Iterator {
    fn collect_to<T>(self, collection: &mut T) -> &mut T
    where
        T: Extend<Self::Item>;
}

impl<I: Iterator> CollectTo for I {
    fn collect_to<T: Extend<I::Item>>(self, collection: &mut T) -> &mut T {
        collection.extend(self);
        collection
    }
}
