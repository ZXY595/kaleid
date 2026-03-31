use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{Data, DataStruct, Error, Field, Ident, Member, Meta, Type, spanned::Spanned};

#[proc_macro_derive(ErrorState, attributes(DoF))]
pub fn derive_error_state(ts: TokenStream) -> TokenStream {
    let derive = syn::parse_macro_input!(ts as syn::DeriveInput);
    let derive_span = derive.span();
    require_struct_input(derive.data, derive_span)
        .and_then(|data| {
            if data.fields.len() > 1 {
                return Ok(delegate_impl_error_state(
                    &derive.ident,
                    data.fields.iter().map(|f| &f.ty),
                    data.fields.members(),
                ));
            }
            let (member, field) = data
                .fields
                .members()
                .zip(data.fields.iter())
                .next()
                .ok_or_else(|| syn::Error::new_spanned(&data.fields, "Expected a field"))?;

            Ok(single_impl_error_state(
                &derive.ident,
                member,
                get_dof(field)?,
            ))
        })
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn require_struct_input(data: Data, span: Span) -> syn::Result<DataStruct> {
    let Data::Struct(data) = data else {
        return Err(Error::new(span, "Expected Struct input"));
    };
    Ok(data)
}

fn get_dof(f: &Field) -> syn::Result<syn::Expr> {
    let span = f.span();
    f.attrs
        .iter()
        .find(|attr| attr.path().is_ident("DoF"))
        .map(|attr| match &attr.meta {
            Meta::List(meta) => meta.parse_args::<syn::Expr>(),
            Meta::NameValue(meta) => Ok(meta.value.clone()),
            Meta::Path(path) => Err(syn::Error::new_spanned(
                path,
                "Expected `DoF = N` or `DoF(N)` where N is a literal integer",
            )),
        })
        .transpose()?
        .ok_or_else(|| syn::Error::new(span, "Expected a field with `#[DoF]` attribute"))
}

fn single_impl_error_state(ident: &Ident, member: Member, dof: syn::Expr) -> TokenStream2 {
    quote! {
        const _: () = {
            use nalgebra::*;
                
            impl ErrorState for #ident {
                type DoF = Const<#dof>;

                fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
                    self.#member.inject(inj)
                }
            }
        };
    }
}

fn delegate_impl_error_state<'a>(
    ident: &Ident,
    tys: impl ExactSizeIterator<Item = &'a Type> + Clone,
    members: impl Iterator<Item = Member>,
) -> TokenStream2 {
    let dof = tys
        .clone()
        .map(|ty| quote!(<#ty as ErrorState>::DoF))
        .reduce(|t1, t2| {
            quote! {
                DimNameSum<#t1, #t2>
            }
        })
        .unwrap_or_else(|| quote!(U0));

    let member_offset = std::iter::once(quote! {0})
        .chain(tys.clone().map(|ty| quote! {<#ty as ErrorState>::DoF::DIM}))
        .take(tys.size_hint().0);

    let member_dim = tys
        .clone()
        .map(|ty| quote! {<#ty as ErrorState>::DoF::name()});

    quote! {
        const _: () = {
            use kaleid::error_state::Members;
            use nalgebra::*;

            impl ErrorState for #ident {
                type DoF = #dof;

                const MEMBERS: Members =
                    Members::new(&[#(Members::single::<#tys>()),*]);

                fn inject(&mut self, inj: Vector<Element, Self::DoF, impl Storage<Element, Self::DoF>>) {
                    #(
                        self.#members.inject(inj.rows_generic(#member_offset, #member_dim));
                    )*
                }
            }
        };
    }
}
