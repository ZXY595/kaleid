//! Extracting specific elements from tuples.
//!
//! Like what the [`frunk`](https://github.com/lloydmeta/frunk) crate implements,
//! but for rust tuples.

/// TODO: Add `diagnostic::on_unimplemented`
pub trait Pluck<Target, Index> {
    /// The elements before the pluck target in the tuple.
    type Before;
    /// The remain elements of the tuple after plucking
    type Remain;

    fn pluck(self) -> (Target, Self::Remain);
}

impl<Target, Tail> Pluck<Target, ()> for (Target, Tail) {
    type Before = ();
    type Remain = Tail;

    fn pluck(self) -> (Target, Self::Remain) {
        self
    }
}

impl<Head, Tail, Target, TailIndex> Pluck<Target, fn() -> TailIndex> for (Head, Tail)
where
    Tail: Pluck<Target, TailIndex>,
{
    type Before = (Head, Tail::Before);
    type Remain = (Head, Tail::Remain);

    fn pluck(self) -> (Target, Self::Remain) {
        let (target, tail) = Tail::pluck(self.1);
        (target, (self.0, tail))
    }
}

impl<'a, Target, Tail> Pluck<&'a Target, ()> for &'a (Target, Tail) {
    type Before = ();
    type Remain = &'a Tail;

    fn pluck(self) -> (&'a Target, Self::Remain) {
        (&self.0, &self.1)
    }
}

impl<'a, Head, Tail, Target, TailIndex> Pluck<&'a Target, fn() -> TailIndex> for &'a (Head, Tail)
where
    &'a Tail: Pluck<&'a Target, TailIndex>,
{
    type Before = (Head, <&'a Tail as Pluck<&'a Target, TailIndex>>::Before);
    type Remain = (&'a Head, <&'a Tail as Pluck<&'a Target, TailIndex>>::Remain);

    fn pluck(self) -> (&'a Target, Self::Remain) {
        let (target, tail) = self.1.pluck();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Pluck<&'a Target, ()> for &'a mut (Target, Tail) {
    type Before = ();
    type Remain = &'a Tail;

    fn pluck(self) -> (&'a Target, Self::Remain) {
        (&self.0, &self.1)
    }
}

impl<'a, Head, Tail, Target, TailIndex> Pluck<&'a Target, fn() -> TailIndex>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Pluck<&'a Target, TailIndex>,
{
    type Before = (Head, <&'a mut Tail as Pluck<&'a Target, TailIndex>>::Before);
    type Remain = (
        &'a Head,
        <&'a mut Tail as Pluck<&'a Target, TailIndex>>::Remain,
    );

    fn pluck(self) -> (&'a Target, Self::Remain) {
        let (target, tail) = self.1.pluck();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Pluck<&'a mut Target, ()> for &'a mut (Target, Tail) {
    type Before = ();
    type Remain = &'a mut Tail;

    fn pluck(self) -> (&'a mut Target, Self::Remain) {
        (&mut self.0, &mut self.1)
    }
}

impl<'a, Head, Tail, Target, TailIndex> Pluck<&'a mut Target, fn() -> TailIndex>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Pluck<&'a mut Target, TailIndex>,
{
    type Before = (
        Head,
        <&'a mut Tail as Pluck<&'a mut Target, TailIndex>>::Before,
    );
    type Remain = (
        &'a mut Head,
        <&'a mut Tail as Pluck<&'a mut Target, TailIndex>>::Remain,
    );

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

impl<THead, TTail, Src, IndexHead, IndexTail> Extract<(THead, TTail), (IndexHead, IndexTail)>
    for Src
where
    Self: Pluck<THead, IndexHead, Remain: Extract<TTail, IndexTail>>,
{
    type Remain = <<Self as Pluck<THead, IndexHead>>::Remain as Extract<TTail, IndexTail>>::Remain;

    #[inline(always)]
    fn extract(self) -> ((THead, TTail), Self::Remain) {
        let (target_head, rest) = self.pluck();
        let (target_tail, remain) = rest.extract();
        ((target_head, target_tail), remain)
    }
}
