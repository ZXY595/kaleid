use derive_deref::{Deref, DerefMut};

#[derive(Deref, DerefMut)]
pub struct Builder<T>(T);

impl<T> Builder<T> {
    pub const fn new() -> Builder<()> {
        Builder(())
    }

    pub fn insert<TT>(self, value: TT) -> Builder<(TT, T)> {
        Builder((value, self.0))
    }

    pub fn build(self) -> T {
        self.0
    }
}
