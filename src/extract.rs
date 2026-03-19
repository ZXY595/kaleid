//! Extracting specific elements from tuples.
//!
//! Like what the [`frunk`](https://github.com/lloydmeta/frunk) crate implements,
//! but for rust tuples.

use crate::DoF;
use nalgebra::DimName;

/// TODO: Add `diagnostic::on_unimplemented`
pub trait Access<Target, Index> {
    /// The remain elements of the tuple after plucking
    type Remain;

    const OFFSET: usize;

    fn pluck(self) -> (Target, Self::Remain);
}

impl<Target, Tail> Access<Target, ()> for (Target, Tail) {
    type Remain = Tail;

    const OFFSET: usize = 0;

    fn pluck(self) -> (Target, Self::Remain) {
        self
    }
}

impl<Head: DoF, Tail, Target, IndexTail> Access<Target, fn() -> IndexTail> for (Head, Tail)
where
    Tail: Access<Target, IndexTail>,
{
    type Remain = (Head, Tail::Remain);

    const OFFSET: usize = Head::DoF::DIM + Tail::OFFSET;

    fn pluck(self) -> (Target, Self::Remain) {
        let (target, tail) = Tail::pluck(self.1);
        (target, (self.0, tail))
    }
}

impl<'a, Target, Tail> Access<&'a Target, ()> for &'a (Target, Tail) {
    type Remain = &'a Tail;

    const OFFSET: usize = 0;

    fn pluck(self) -> (&'a Target, Self::Remain) {
        (&self.0, &self.1)
    }
}

impl<'a, Head: DoF, Tail, Target, IndexTail> Access<&'a Target, fn() -> IndexTail>
    for &'a (Head, Tail)
where
    &'a Tail: Access<&'a Target, IndexTail>,
{
    type Remain = (
        &'a Head,
        <&'a Tail as Access<&'a Target, IndexTail>>::Remain,
    );

    const OFFSET: usize = Head::DoF::DIM + <&Tail>::OFFSET;

    fn pluck(self) -> (&'a Target, Self::Remain) {
        let (target, tail) = self.1.pluck();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Access<&'a Target, ()> for &'a mut (Target, Tail) {
    type Remain = &'a Tail;

    const OFFSET: usize = 0;

    fn pluck(self) -> (&'a Target, Self::Remain) {
        (&self.0, &self.1)
    }
}

impl<'a, Head: DoF, Tail, Target, IndexTail> Access<&'a Target, fn() -> IndexTail>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Access<&'a Target, IndexTail>,
{
    type Remain = (
        &'a Head,
        <&'a mut Tail as Access<&'a Target, IndexTail>>::Remain,
    );

    const OFFSET: usize = Head::DoF::DIM + <&mut Tail>::OFFSET;

    fn pluck(self) -> (&'a Target, Self::Remain) {
        let (target, tail) = self.1.pluck();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Access<&'a mut Target, ()> for &'a mut (Target, Tail) {
    type Remain = &'a mut Tail;

    const OFFSET: usize = 0;

    fn pluck(self) -> (&'a mut Target, Self::Remain) {
        (&mut self.0, &mut self.1)
    }
}

impl<'a, Head: DoF, Tail, Target, IndexTail> Access<&'a mut Target, fn() -> IndexTail>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Access<&'a mut Target, IndexTail>,
{
    type Remain = (
        &'a mut Head,
        <&'a mut Tail as Access<&'a mut Target, IndexTail>>::Remain,
    );

    const OFFSET: usize = Head::DoF::DIM + <&mut Tail>::OFFSET;

    fn pluck(self) -> (&'a mut Target, Self::Remain) {
        let (target, tail) = self.1.pluck();
        (target, (&mut self.0, tail))
    }
}

pub trait Extract<Target, Indices> {
    type Remain;

    fn extract(self) -> (Target, Self::Remain);
}

impl<Src> Extract<(), ()> for Src {
    type Remain = Src;

    #[inline(always)]
    fn extract(self) -> ((), Self::Remain) {
        ((), self)
    }
}

impl<TargetHead, TargetTail, Src, IndexHead, IndexTail>
    Extract<(TargetHead, TargetTail), (IndexHead, IndexTail)> for Src
where
    Self: Access<TargetHead, IndexHead, Remain: Extract<TargetTail, IndexTail>>,
{
    type Remain =
        <<Self as Access<TargetHead, IndexHead>>::Remain as Extract<TargetTail, IndexTail>>::Remain;

    #[inline(always)]
    fn extract(self) -> ((TargetHead, TargetTail), Self::Remain) {
        let (target_head, rest) = self.pluck();
        let (target_tail, remain) = rest.extract();
        ((target_head, target_tail), remain)
    }
}
