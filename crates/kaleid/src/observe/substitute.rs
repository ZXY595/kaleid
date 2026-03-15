use nalgebra::{Cholesky, ComplexField, DefaultAllocator, Dim, OMatrix, allocator::Allocator};

/// A type that can provides a positive definite substitute value
/// for some inverse operations like [`Cholesky::new_with_substitute`]
///
/// # SAFETY
///
/// the value of [`Substitutive::substitute`] must return a positive definite value
pub(crate) unsafe trait Substitutive: ComplexField {
    fn substitute() -> Self;

    fn non_zero_or_substitute(self) -> Self {
        if self.is_zero() {
            Self::substitute()
        } else {
            self
        }
    }
}

/// # SAFETY:
///
/// returned value is positive definite
unsafe impl<T: ComplexField> Substitutive for T {
    #[inline]
    fn substitute() -> Self {
        nalgebra::convert(0.0001)
    }
}

pub(crate) trait InverseWithSubstitute {
    fn cholesky_inverse_with_substitute(self) -> Self;
}

impl<T: ComplexField, D: Dim> InverseWithSubstitute for OMatrix<T, D, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    fn cholesky_inverse_with_substitute(self) -> Self {
        let cholesky = Cholesky::new_with_substitute(self, T::substitute());
        // # SAFETY:
        //
        // this is safe because the value of `T::SUBSTITUTE` is positive definite
        // and the Cholesky decomposition is always successful
        let cholesky = unsafe { cholesky.unwrap_unchecked() };
        cholesky.inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::*;
    fn is_positive_definite(x: impl ComplexField) -> bool {
        ComplexField::try_sqrt(x).is_some()
    }

    #[test]
    fn test_substitute_positive_definite() {
        assert!(is_positive_definite(f64::substitute()));
        assert!(is_positive_definite(f32::substitute()));
    }
}
