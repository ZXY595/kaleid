//! Extracting specific elements from tuples.
//!
//! Like what the [`frunk`](https://github.com/lloydmeta/frunk) crate implements,
//! but for rust tuples.

use crate::DoF;
use nalgebra::DimName;

/// TODO: Add `diagnostic::on_unimplemented`
pub trait Extract<Target, Index, Mode: ExtractMode = Mono> {
    /// The remain elements of the tuple after plucking
    type Remain;

    const OFFSET: Mode::Offset;

    fn extract(self) -> (Target, Self::Remain);
}

pub trait ExtractMode {
    type Offset: Into<Option<usize>>;
}

pub struct Mono;
pub struct Multi;

impl ExtractMode for Mono {
    type Offset = usize;
}

impl ExtractMode for Multi {
    type Offset = Option<usize>;
}

impl<Target, Tail> Extract<Target, ()> for (Target, Tail) {
    type Remain = Tail;

    const OFFSET: usize = 0;

    fn extract(self) -> (Target, Self::Remain) {
        self
    }
}

impl<Head: DoF, Tail, Target, IndexTail> Extract<Target, fn() -> IndexTail> for (Head, Tail)
where
    Tail: Extract<Target, IndexTail>,
{
    type Remain = (Head, Tail::Remain);

    const OFFSET: usize = Head::DoF::DIM + Tail::OFFSET;

    fn extract(self) -> (Target, Self::Remain) {
        let (target, tail) = Tail::extract(self.1);
        (target, (self.0, tail))
    }
}

impl<'a, Target, Tail> Extract<&'a Target, ()> for &'a (Target, Tail) {
    type Remain = &'a Tail;

    const OFFSET: usize = 0;

    fn extract(self) -> (&'a Target, Self::Remain) {
        (&self.0, &self.1)
    }
}

impl<'a, Head: DoF, Tail, Target, IndexTail> Extract<&'a Target, fn() -> IndexTail>
    for &'a (Head, Tail)
where
    &'a Tail: Extract<&'a Target, IndexTail>,
{
    type Remain = (
        &'a Head,
        <&'a Tail as Extract<&'a Target, IndexTail>>::Remain,
    );

    const OFFSET: usize = Head::DoF::DIM + <&Tail>::OFFSET;

    fn extract(self) -> (&'a Target, Self::Remain) {
        let (target, tail) = (&self.1).extract();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Extract<&'a Target, ()> for &'a mut (Target, Tail) {
    type Remain = &'a Tail;

    const OFFSET: usize = 0;

    fn extract(self) -> (&'a Target, Self::Remain) {
        (&self.0, &self.1)
    }
}

impl<'a, Head: DoF, Tail, Target, IndexTail> Extract<&'a Target, fn() -> IndexTail>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Extract<&'a Target, IndexTail>,
{
    type Remain = (
        &'a Head,
        <&'a mut Tail as Extract<&'a Target, IndexTail>>::Remain,
    );

    const OFFSET: usize = Head::DoF::DIM + <&mut Tail>::OFFSET;

    fn extract(self) -> (&'a Target, Self::Remain) {
        let (target, tail) = (&mut self.1).extract();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Extract<&'a mut Target, ()> for &'a mut (Target, Tail) {
    type Remain = &'a mut Tail;

    const OFFSET: usize = 0;

    fn extract(self) -> (&'a mut Target, Self::Remain) {
        (&mut self.0, &mut self.1)
    }
}

impl<'a, Head: DoF, Tail, Target, IndexTail> Extract<&'a mut Target, fn() -> IndexTail>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Extract<&'a mut Target, IndexTail>,
{
    type Remain = (
        &'a mut Head,
        <&'a mut Tail as Extract<&'a mut Target, IndexTail>>::Remain,
    );

    const OFFSET: usize = Head::DoF::DIM + <&mut Tail>::OFFSET;

    fn extract(self) -> (&'a mut Target, Self::Remain) {
        let (target, tail) = (&mut self.1).extract();
        (target, (&mut self.0, tail))
    }
}

#[diagnostic::do_not_recommend]
impl<Src> Extract<(), (), Multi> for Src {
    type Remain = Src;

    const OFFSET: Option<usize> = Some(0);

    #[inline(always)]
    fn extract(self) -> ((), Self::Remain) {
        ((), self)
    }
}

impl<TargetHead: DoF, TargetTail, Src, IndexHead, IndexTail>
    Extract<(TargetHead, TargetTail), (IndexHead, IndexTail), Multi> for Src
where
    Self: Extract<TargetHead, IndexHead, Remain: Extract<TargetTail, IndexTail, Multi>>,
{
    type Remain = <<Self as Extract<TargetHead, IndexHead>>::Remain as Extract<
        TargetTail,
        IndexTail,
        Multi,
    >>::Remain;

    const OFFSET: Option<usize> = {
        let head_offset = <Self as Extract<TargetHead, IndexHead>>::OFFSET;
        let tail_offset = <Self as Extract<TargetHead, IndexHead>>::Remain::OFFSET;
        match tail_offset {
            Some(offset) if head_offset + TargetHead::DoF::DIM == offset => Some(head_offset),
            Some(offset) if offset + TargetHead::DoF::DIM == head_offset => Some(offset),
            _ => None,
        }
    };

    #[inline(always)]
    fn extract(self) -> ((TargetHead, TargetTail), Self::Remain) {
        let (target_head, rest) = self.extract();
        let (target_tail, remain) = rest.extract();
        ((target_head, target_tail), remain)
    }
}

#[diagnostic::do_not_recommend]
impl<TS: Tuples, Src, Indices> Extract<TS, (Indices,), Multi> for Src
where
    Src: Extract<TS::List, Indices, Multi>,
{
    type Remain = <Src as Extract<TS::List, Indices, Multi>>::Remain;

    const OFFSET: Option<usize> = <Src as Extract<TS::List, Indices, Multi>>::OFFSET;

    fn extract(self) -> (TS, Self::Remain) {
        let (ts, remain) = self.extract();
        (TS::from_tuple_list(ts), remain)
    }
}

pub trait Tuples {
    type List;
    fn to_tuple_list(self) -> Self::List;
    fn from_tuple_list(list: Self::List) -> Self;
}

impl<T1, T2> Tuples for (T1, T2) {
    type List = (T1, (T2, ()));

    fn to_tuple_list(self) -> Self::List {
        let (t1, t2) = self;
        (t1, (t2, ()))
    }

    fn from_tuple_list((t1, (t2, _)): Self::List) -> Self {
        (t1, t2)
    }
}

impl<T1, T2, T3> Tuples for (T1, T2, T3) {
    type List = (T1, (T2, (T3, ())));

    fn to_tuple_list(self) -> Self::List {
        let (t1, t2, t3) = self;
        (t1, (t2, (t3, ())))
    }

    fn from_tuple_list((t1, (t2, (t3, _))): Self::List) -> Self {
        (t1, t2, t3)
    }
}
