pub(crate) trait Tuples {
    type Lists;
    fn to_lists(self) -> Self::Lists;
    fn from_lists(lists: Self::Lists) -> Self;
}

macro_rules! impl_ {
    {
        ($($ident:ident: $ty:ident),*) => $pat:tt in $list_ty:ty
    } => {
        impl<$($ty),*> Tuples for ($($ty),*) {
            type Lists = $list_ty;

            fn to_lists(self) -> Self::Lists {
                let ($($ident),*) = self;
                $pat
            }

            fn from_lists($pat: Self::Lists) -> Self {
                ($($ident),*)
            }
        }
    };
}

impl_! {(t1: T1, t2: T2, t3: T3) => (t1, (t2, t3)) in (T1, (T2, T3))}
impl_! {(t1: T1, t2: T2, t3: T3, t4: T4) => (t1, (t2, (t3, t4))) in (T1, (T2, (T3, T4)))}
impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5) => (t1, (t2, (t3, (t4, t5)))) in (T1, (T2, (T3, (T4, T5))))}
impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6) => (t1, (t2, (t3, (t4, (t5, t6))))) in (T1, (T2, (T3, (T4, (T5, T6)))))}
impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7) => (t1, (t2, (t3, (t4, (t5, (t6, t7)))))) in (T1, (T2, (T3, (T4, (T5, (T6, T7))))))}
impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, t8))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, T8)))))))}
impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8, t9: T9) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, (t8, t9)))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, (T8, T9))))))))}
// impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8, t9: T9, t10: T10) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, (t8, (t9, t10))))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, (T8, (T9, T10)))))))))}
// impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8, t9: T9, t10: T10, t11: T11) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, (t8, (t9, (t10, t11)))))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, (T8, (T9, (T10, T11))))))))))}
// impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8, t9: T9, t10: T10, t11: T11, t12: T12) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, (t8, (t9, (t10, (t11, t12))))))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, (T8, (T9, (T10, (T11, T12)))))))))))}
// impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8, t9: T9, t10: T10, t11: T11, t12: T12, t13: T13) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, (t8, (t9, (t10, (t11, (t12, t13)))))))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, (T8, (T9, (T10, (T11, (T12, T13))))))))))))}
// impl_! {(t1: T1, t2: T2, t3: T3, t4: T4, t5: T5, t6: T6, t7: T7, t8: T8, t9: T9, t10: T10, t11: T11, t12: T12, t13: T13, t14: T14) => (t1, (t2, (t3, (t4, (t5, (t6, (t7, (t8, (t9, (t10, (t11, (t12, (t13, t14))))))))))))) in (T1, (T2, (T3, (T4, (T5, (T6, (T7, (T8, (T9, (T10, (T11, (T12, (T13, T14)))))))))))))}
