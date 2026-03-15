use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{Data, DataStruct, Error, Field, Fields, Ident, Member, Meta, spanned::Spanned};

#[proc_macro_derive(ErrorState, attributes(DoF))]
pub fn derive_error_state(ts: TokenStream) -> TokenStream {
    let derive = syn::parse_macro_input!(ts as syn::DeriveInput);
    let derive_span = derive.span();
    require_struct_input(derive.data, || derive_span)
        .and_then(|data| {
            let member = data
                .fields
                .members()
                .next()
                .ok_or_else(|| syn::Error::new_spanned(&data.fields, "Expected a field"))?;

            let field = require_single_field(data.fields)?;
            let dof = get_dof(field)?;
            Ok(impl_error_state(&derive.ident, member, dof))
        })
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn require_struct_input(data: Data, span: impl FnOnce() -> Span) -> syn::Result<DataStruct> {
    let Data::Struct(data) = data else {
        return Err(Error::new(span(), "Expected Struct input"));
    };
    Ok(data)
}

fn require_single_field(fields: Fields) -> syn::Result<Field> {
    let span = fields.span();
    if fields.len() > 1 {
        return Err(Error::new(span, "Expected a single field"));
    }
    fields
        .into_iter()
        .next()
        .ok_or_else(|| Error::new(span, "Expected a field"))
}

fn get_dof(f: Field) -> syn::Result<syn::Expr> {
    let span = f.span();
    f.attrs
        .into_iter()
        .find(|attr| attr.path().is_ident("DoF"))
        .map(|attr| match attr.meta {
            Meta::List(meta) => meta.parse_args::<syn::Expr>(),
            Meta::NameValue(meta) => Ok(meta.value),
            Meta::Path(path) => Err(syn::Error::new_spanned(
                &path,
                "Expected `DoF = N` or `DoF(N)` where N is a literal integer",
            )),
        })
        .transpose()?
        .ok_or_else(|| syn::Error::new(span, "Expected a field with `#[DoF]` attribute"))
}

fn impl_error_state(ident: &Ident, member: Member, dof: syn::Expr) -> TokenStream2 {
    quote! {
        impl kaleid::DoF for #ident {
            type DoF = Const<#dof>;
        }

        impl ErrorState for #ident {
            fn inject(&mut self, inj: nalgebra::Vector<Element, Self::DoF, impl nalgebra::Storage<Element, Self::DoF>>) {
                self.#member.inject(inj)
            }
        }
    }
}
