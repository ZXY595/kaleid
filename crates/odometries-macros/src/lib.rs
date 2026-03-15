pub(crate) mod kf_state;
pub(crate) mod state_offset;
pub(crate) mod utils;
use proc_macro::TokenStream;
use quote::ToTokens;

/// Derive macro for `KFState`, also implements `SubStateOf<this struct>` for every field.
#[proc_macro_derive(KFState, attributes(element))]
pub fn derive_kf_state(ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as kf_state::KFState)
        .to_token_stream()
        .into()
}

/// # Example
///
/// ```rust
/// struct State<T> {
///     state1: State1,
/// }
/// #[sub_state_of(State)]
/// struct State1<T: Scalar>(State2<T>, State3<T>);
/// ```
#[proc_macro_attribute]
pub fn sub_state_of(mut arg: TokenStream, ts: TokenStream) -> TokenStream {
    arg.extend(ts);
    let ts = arg;
    syn::parse_macro_input!(ts as state_offset::Input)
        .to_token_stream()
        .into()
}
