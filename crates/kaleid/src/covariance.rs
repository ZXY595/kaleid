use nalgebra::{Const, DimName, DimNameAdd, DimNameSum};
use nalgebra::{MatrixView, MatrixViewMut, OMatrix, U1};

use crate::state::bundle::{AccessSet, AccessStatus, reduce_access};
use crate::state::{ConstArray, IntoArray};
use crate::tuples::Tuples;
use crate::{
    KFState,
    state::bundle::{Access, AsMember, StateBundle},
};

pub type Cov<S: KFState> = OMatrix<S::Element, S::Dim, S::Dim>;

pub struct CovView<'cov, B: StateBundle, Rows: MembersOf<B>, Cols: MembersOf<B> = Rows>(
    pub MatrixView<'cov, B::Element, Rows::Dim, Cols::Dim, U1, B::Dim>,
);

pub struct CovViewMut<'cov, B: StateBundle, Rows: MembersOf<B>, Cols: MembersOf<B> = Rows>(
    pub MatrixViewMut<'cov, B::Element, Rows::Dim, Cols::Dim, U1, B::Dim>,
);

pub type CovRowsView<'cov, B, Rows> = CovView<'cov, B, Rows, AllMembers>;
pub type CovColsView<'cov, B, Cols> = CovView<'cov, B, AllMembers, Cols>;

struct AccessTable<N: IntoArray>(ConstArray<ConstArray<AccessStatus, N>, N>);

impl<const N: usize> AccessTable<Const<N>> {
    const fn new() -> Self {
        Self([[None; N]; N])
    }

    const fn from_row(val: [AccessStatus; N]) -> Self {
        let mut temp = Self::new();
        while let mut row = 0
            && let Some(row) = crate::incr_until(&mut row, N)
        {
            temp.0[row] = val;
        }
        temp
    }

    const fn from_col(val: [AccessStatus; N]) -> Self {
        let mut temp = Self::new();
        while let mut row = 0
            && let Some(row) = crate::incr_until(&mut row, N)
        {
            while let mut col = 0
                && let Some(col) = crate::incr_until(&mut col, N)
            {
                temp.0[row][col] = val[col];
            }
        }
        temp
    }

    const fn reduce(mut self, rhs: Self) -> Self {
        while let mut row = 0
            && let Some(row) = crate::incr_until(&mut row, N)
        {
            while let mut col = 0
                && let Some(col) = crate::incr_until(&mut col, N)
            {
                self.0[row][col] = reduce_access(self.0[row][col], rhs.0[row][col]);
            }
        }
        self
    }
}

pub trait FromCov<'cov, Bundle: StateBundle>: Sized {
    const MASK: AccessTable<Bundle::Dim>;

    fn from_cov(
        cov: &'cov mut MatrixViewMut<'cov, Bundle::Element, Bundle::Dim, Bundle::Dim>,
    ) -> Self;
}

pub(crate) trait MembersOf<B: StateBundle> {
    type Dim: DimName;
    const START_MEMBER: usize;
    const START: usize = fold_start(B::DIMS, Self::START_MEMBER);
    const MASK: AccessSet<B::Dim>;
}

pub(crate) struct AllMembers;

const fn fold_start(dims: &[usize], member: usize) -> usize {
    let mut start = 0;

    while let mut i = 0
        && let Some(i) = crate::incr_until(&mut i, member)
    {
        start += dims[i];
    }
    start
}

impl<T, B, const N: usize> MembersOf<B> for T
where
    T: KFState,
    B: StateBundle<Dim = Const<N>> + AsMember<T>,
{
    type Dim = T::Dim;
    const START_MEMBER: usize = B::MEMBER;
    const MASK: AccessSet<Const<N>> = AccessSet::new().with_reference(B::MEMBER, Access::Write);
}

impl<B, const N: usize> MembersOf<B> for AllMembers
where
    B: StateBundle<Dim = Const<N>>,
{
    type Dim = B::Dim;

    const START_MEMBER: usize = 0;
    const START: usize = 0;
    const MASK: AccessSet<Const<N>> = AccessSet::all_some(Access::Write);
}

impl<T1, T2, B, const N: usize> MembersOf<B> for (T1, T2)
where
    B: StateBundle<Dim = Const<N>>,
    T1: MembersOf<B, Dim: DimNameAdd<T2::Dim>>,
    T2: MembersOf<B>,
{
    type Dim = DimNameSum<T1::Dim, T2::Dim>;
    const START_MEMBER: usize = {
        assert!(
            T2::START_MEMBER - T1::START_MEMBER == 1,
            "Incontiguous members detected"
        );
        T1::START_MEMBER
    };
    const MASK: AccessSet<Const<N>> = T1::MASK.reduce(T2::MASK);
}

impl<B, TS> MembersOf<B> for TS
where
    B: StateBundle,
    TS: Tuples<Lists: MembersOf<B>>,
{
    type Dim = <TS::Lists as MembersOf<B>>::Dim;

    const START_MEMBER: usize = TS::Lists::START_MEMBER;

    const MASK: AccessSet<B::Dim> = TS::Lists::MASK;
}

impl<'cov, Bundle, const N: usize, Rows, Cols> FromCov<'cov, Bundle>
    for CovViewMut<'cov, Bundle, Rows, Cols>
where
    Bundle: StateBundle<Dim = Const<N>>,
    Rows: MembersOf<Bundle>,
    Cols: MembersOf<Bundle>,
{
    const MASK: AccessTable<Const<N>> =
        AccessTable::from_row(Cols::MASK.0).reduce(AccessTable::from_col(Rows::MASK.0));

    fn from_cov(
        cov: &'cov mut MatrixViewMut<'cov, Bundle::Element, Bundle::Dim, Bundle::Dim>,
    ) -> Self {
        let view = cov.generic_view_mut(
            const { (Rows::START, Cols::START) },
            (Rows::Dim::name(), Cols::Dim::name()),
        );
        Self(view)
    }
}

impl<const N: usize> AccessTable<Const<N>> {
    const fn with_all_reference_immutable(mut self) -> Self {
        while let mut row = 0
            && let Some(row) = crate::incr_until(&mut row, N)
        {
            while let mut col = 0
                && let Some(col) = crate::incr_until(&mut col, N)
            {
                self.0[row][col] = match self.0[row][col] {
                    Some(_) => Some(Access::Read),
                    None => None,
                };
            }
        }
        self
    }
}

impl<'cov, Bundle, const N: usize, Rows, Cols> FromCov<'cov, Bundle>
    for CovView<'cov, Bundle, Rows, Cols>
where
    Bundle: StateBundle<Dim = Const<N>>,
    Rows: MembersOf<Bundle>,
    Cols: MembersOf<Bundle>,
{
    const MASK: AccessTable<Const<N>> = AccessTable::from_row(Cols::MASK.0)
        .reduce(AccessTable::from_col(Rows::MASK.0))
        .with_all_reference_immutable();

    fn from_cov(
        cov: &'cov mut MatrixViewMut<'cov, Bundle::Element, Bundle::Dim, Bundle::Dim>,
    ) -> Self {
        let view = cov.generic_view(
            const { (Rows::START, Cols::START) },
            (Rows::Dim::name(), Cols::Dim::name()),
        );
        Self(view)
    }
}

impl<'cov, Bundle, const N: usize, T1, T2> FromCov<'cov, Bundle> for (T1, T2)
where
    Bundle: StateBundle<Dim = Const<N>>,
    T1: FromCov<'cov, Bundle>,
    T2: FromCov<'cov, Bundle>,
{
    const MASK: AccessTable<Const<N>> = T1::MASK.reduce(T2::MASK);

    fn from_cov(
        cov: &'cov mut MatrixViewMut<'cov, Bundle::Element, Bundle::Dim, Bundle::Dim>,
    ) -> Self {
        let ptr = &raw mut *cov;
        // Safety:
        // const eval ensures that there's no overlapping between T1 and T2
        unsafe { (T1::from_cov(&mut *ptr), T2::from_cov(&mut *ptr)) }
    }
}

impl<'cov, Bundle, const N: usize, TS> FromCov<'cov, Bundle> for TS
where
    Bundle: StateBundle<Dim = Const<N>>,
    TS: Tuples<Lists: FromCov<'cov, Bundle>>,
{
    const MASK: AccessTable<Bundle::Dim> = TS::Lists::MASK;

    fn from_cov(
        cov: &'cov mut MatrixViewMut<'cov, Bundle::Element, Bundle::Dim, Bundle::Dim>,
    ) -> Self {
        Self::from_lists(TS::Lists::from_cov(cov))
    }
}
