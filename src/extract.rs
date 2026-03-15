//! Extracting specific elements from tuples.
//!
//! Like what the [`frunk`](https://github.com/lloydmeta/frunk) crate implements,
//! but for rust tuples.

pub trait Pluck<Target, Index> {
    type Before;
    type Remaining;

    fn pluck(self) -> (Target, Self::Remaining);
}

impl<Target, Tail> Pluck<Target, ()> for (Target, Tail) {
    type Before = ();
    type Remaining = Tail;

    fn pluck(self) -> (Target, Self::Remaining) {
        self
    }
}

impl<Head, Tail, Target, TailIndex> Pluck<Target, fn() -> TailIndex> for (Head, Tail)
where
    Tail: Pluck<Target, TailIndex>,
{
    type Before = (Head, Tail::Before);
    type Remaining = (Head, Tail::Remaining);

    fn pluck(self) -> (Target, Self::Remaining) {
        let (target, tail) = Tail::pluck(self.1);
        (target, (self.0, tail))
    }
}

impl<'a, Target, Tail> Pluck<&'a Target, ()> for &'a (Target, Tail) {
    type Before = ();
    type Remaining = &'a Tail;

    fn pluck(self) -> (&'a Target, Self::Remaining) {
        (&self.0, &self.1)
    }
}

impl<'a, Head, Tail, Target, TailIndex> Pluck<&'a Target, fn() -> TailIndex> for &'a (Head, Tail)
where
    &'a Tail: Pluck<&'a Target, TailIndex>,
{
    type Before = (Head, <&'a Tail as Pluck<&'a Target, TailIndex>>::Before);
    type Remaining = (
        &'a Head,
        <&'a Tail as Pluck<&'a Target, TailIndex>>::Remaining,
    );

    fn pluck(self) -> (&'a Target, Self::Remaining) {
        let (target, tail) = self.1.pluck();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Pluck<&'a Target, ()> for &'a mut (Target, Tail) {
    type Before = ();
    type Remaining = &'a Tail;

    fn pluck(self) -> (&'a Target, Self::Remaining) {
        (&self.0, &self.1)
    }
}

impl<'a, Head, Tail, Target, TailIndex> Pluck<&'a Target, fn() -> TailIndex>
    for &'a mut (Head, Tail)
where
    &'a mut Tail: Pluck<&'a Target, TailIndex>,
{
    type Before = (Head, <&'a mut Tail as Pluck<&'a Target, TailIndex>>::Before);
    type Remaining = (
        &'a Head,
        <&'a mut Tail as Pluck<&'a Target, TailIndex>>::Remaining,
    );

    fn pluck(self) -> (&'a Target, Self::Remaining) {
        let (target, tail) = self.1.pluck();
        (target, (&self.0, tail))
    }
}

impl<'a, Target, Tail> Pluck<&'a mut Target, ()> for &'a mut (Target, Tail) {
    type Before = ();
    type Remaining = &'a mut Tail;

    fn pluck(self) -> (&'a mut Target, Self::Remaining) {
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
    type Remaining = (
        &'a mut Head,
        <&'a mut Tail as Pluck<&'a mut Target, TailIndex>>::Remaining,
    );

    fn pluck(self) -> (&'a mut Target, Self::Remaining) {
        let (target, tail) = self.1.pluck();
        (target, (&mut self.0, tail))
    }
}

pub trait Extract<Target, Indices> {
    type Remaining;

    fn extract(self) -> (Target, Self::Remaining);
}

impl<Src> Extract<(), ()> for Src {
    type Remaining = Src;

    #[inline(always)]
    fn extract(self) -> ((), Self::Remaining) {
        ((), self)
    }
}

impl<THead, TTail, Src, IndexHead, IndexTail> Extract<(THead, TTail), (IndexHead, IndexTail)>
    for Src
where
    Self: Pluck<THead, IndexHead, Remaining: Extract<TTail, IndexTail>>,
{
    type Remaining =
        <<Self as Pluck<THead, IndexHead>>::Remaining as Extract<TTail, IndexTail>>::Remaining;

    #[inline(always)]
    fn extract(self) -> ((THead, TTail), Self::Remaining) {
        let (target_head, rest) = self.pluck();
        let (target_tail, remain) = rest.extract();
        ((target_head, target_tail), remain)
    }
}
