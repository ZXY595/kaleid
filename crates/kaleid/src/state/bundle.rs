use nalgebra::Const;

use crate::{
    KFState,
    state::{ConstArray, IntoArray},
};

pub trait StateBundle: KFState {
    const DIMS: &'static [usize];
}

pub trait AsMember<T> {
    const MEMBER: usize;
}

pub trait FromBundle<'b, Bundle: StateBundle>: Sized {
    const MASK: AccessSet<Bundle::Dim>;

    fn from_bundle(bundle: &'b mut Bundle) -> Self;
}
// pub type BundleMask =

pub(crate) struct AccessSet<N: IntoArray>(pub(crate) ConstArray<AccessStatus, N>);

impl<const N: usize> AccessSet<Const<N>> {
    pub const fn new() -> Self {
        Self([None; N])
    }

    pub const fn all_some(reference: Access) -> Self {
        Self([Some(reference); N])
    }

    pub const fn index_mut(&mut self, index: usize) -> &mut AccessStatus {
        &mut self.0[index]
    }

    pub const fn with_reference(mut self, index: usize, reference: Access) -> Self {
        self.0[index] = Some(reference);
        self
    }

    /// Reduce two masks into a single mask.
    ///
    /// # Panic
    /// If the two borrow masks have overlapping mutable references.
    pub const fn reduce(mut self, rhs: Self) -> Self {
        while let mut i = 0
            && let Some(i) = crate::incr_until(&mut i, N)
        {
            self.0[i] = reduce_access(self.0[i], rhs.0[i]);
        }
        self
    }
}

pub(crate) const fn reduce_access(lhs: AccessStatus, rhs: AccessStatus) -> AccessStatus {
    match (lhs, rhs) {
        (None, None) => None,
        (Some(reference), None) | (None, Some(reference)) => Some(reference),
        (Some(Access::Read), Some(Access::Read)) => Some(Access::Read),
        (Some(Access::Write), Some(_)) | (Some(_), Some(Access::Write)) => {
            panic!("Cannot borrow mutable reference more than once at a time")
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Access {
    Read,
    Write,
}

pub(crate) type AccessStatus = Option<Access>;

pub const fn bundle_member<B, M>() -> usize
where
    B: AsMember<M>,
{
    B::MEMBER
}

impl<'b, Bundle, const N: usize, Member> FromBundle<'b, Bundle> for &'b Member
where
    Bundle: StateBundle<Dim = Const<N>> + AsMember<Member> + AsRef<Member>,
{
    const MASK: AccessSet<Const<N>> = AccessSet::new().with_reference(Bundle::MEMBER, Access::Read);

    fn from_bundle(bundle: &'b mut Bundle) -> Self {
        AsRef::as_ref(bundle)
    }
}

impl<'b, Bundle, const N: usize, Member> FromBundle<'b, Bundle> for &'b mut Member
where
    Bundle: StateBundle<Dim = Const<N>> + AsMember<Member> + AsMut<Member>,
{
    const MASK: AccessSet<Const<N>> =
        AccessSet::new().with_reference(Bundle::MEMBER, Access::Write);

    fn from_bundle(bundle: &'b mut Bundle) -> Self {
        bundle.as_mut()
    }
}

impl<'b, const N: usize, Bundle, T1, T2> FromBundle<'b, Bundle> for (T1, T2)
where
    Bundle: StateBundle<Dim = Const<N>>,
    T1: FromBundle<'b, Bundle>,
    T2: FromBundle<'b, Bundle>,
{
    const MASK: AccessSet<Const<N>> = T1::MASK.reduce(T2::MASK);

    #[inline]
    fn from_bundle(bundle: &'b mut Bundle) -> Self {
        let ptr = &raw mut *bundle;
        // Safety:
        // const eval ensures that there's no overlapping between T1 and T2
        unsafe { (T1::from_bundle(&mut *ptr), T2::from_bundle(&mut *ptr)) }
    }
}
