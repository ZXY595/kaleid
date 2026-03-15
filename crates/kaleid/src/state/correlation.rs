use crate::state::{
    StateDim,
    common::{BiasState, State, StateWithBias},
};
use odometries_core::ImplMatrix;

use nalgebra::{ClosedAddAssign, DefaultAllocator, Dim, DimName, Scalar, allocator::Allocator};

use super::{KFState, StateOffset};

// /// A trait for a sub-state that can be correlated to a super-state.
// pub trait CorrelateTo<Super: KFState>: StateOffset<Super> {
//     type CorDim: DimName;
//
//     fn correlate_to<D: Dim>(
//         s: &ImplMatrix!(Super::Element, D, Super::Dim),
//     ) -> ImplMatrix!(Super::Element, D, Self::CorDim)
//     where
//         DefaultAllocator: Allocator<D, Self::CorDim>;
//
//     fn correlate_from<D: Dim>(
//         s: &ImplMatrix!(Super::Element, Super::Dim, D),
//     ) -> ImplMatrix!(Super::Element, Self::CorDim, D)
//     where
//         DefaultAllocator: Allocator<Self::CorDim, D>;
// }
//
// pub(crate) type SensitivityDim<S, Super> = <S as CorrelateTo<Super>>::CorDim;
//
// /// A [`KFState`] with no inner bias sub-state.
// /// but the [`KFState`] that all sub-states are [`Unbiased`] is also considered to be [`Unbiased`].
// pub trait Unbiased {}
//
// impl<S, M, B, Super: KFState> CorrelateTo<Super> for CommonState<S, M, UnBias<B>>
// where
//     Self: StateOffset<Super, Element = Super::Element>,
// {
//     type CorDim = Self::Dim;
//
//     #[inline]
//     fn correlate_to<D: Dim>(
//         s: &ImplMatrix!(Super::Element, D, Super::Dim),
//     ) -> ImplMatrix!(Super::Element, D, Self::CorDim)
//     where
//         DefaultAllocator: Allocator<D, Self::CorDim>,
//     {
//         s.columns_generic(Self::Offset::DIM, Self::CorDim::name())
//     }
//
//     #[inline]
//     fn correlate_from<D: Dim>(
//         s: &ImplMatrix!(Super::Element, Super::Dim, D),
//     ) -> ImplMatrix!(Super::Element, Self::CorDim, D)
//     where
//         DefaultAllocator: Allocator<Self::CorDim, D>,
//     {
//         s.rows_generic(Self::Offset::DIM, Self::CorDim::name())
//     }
// }
//
// impl<S, M, B, Super> CorrelateTo<Super> for MarkedStateWithBias<S, M, B>
// where
//     Super: KFState<Element: Scalar + ClosedAddAssign>,
//     Super: StateOffset<Self, Element = Self::Element>,
//     CommonState<S, M>: StateOffset<Self, Element = Self::Element>,
//     BiasState<S, B>: StateOffset<Self, Element = Self::Element, Dim = StateDim<CommonState<S, M>>>,
// {
//     type CorDim = StateDim<CommonState<S, M>>;
//
//     #[inline]
//     fn correlate_to<D: Dim>(
//         s: &ImplMatrix!(Super::Element, D, Super::Dim),
//     ) -> ImplMatrix!(Super::Element, D, Self::CorDim)
//     where
//         DefaultAllocator: Allocator<D, Self::CorDim>,
//     {
//         let s = s.columns_generic(Self::Offset::DIM, Self::Dim::name());
//         CommonState::<S, M>::correlate_to(&s) + BiasState::<S, B>::correlate_to(&s)
//     }
//
//     #[inline]
//     fn correlate_from<D: Dim>(
//         s: &ImplMatrix!(Super::Element, Super::Dim, D),
//     ) -> ImplMatrix!(Super::Element, Self::CorDim, D)
//     where
//         DefaultAllocator: Allocator<Self::CorDim, D>,
//     {
//         let s = s.rows_generic(Self::Offset::DIM, Self::Dim::name());
//         CommonState::<S, M>::correlate_from(&s) + BiasState::<S, B>::correlate_from(&s)
//     }
// }
