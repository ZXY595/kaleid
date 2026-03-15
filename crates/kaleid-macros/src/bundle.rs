use extension_traits::extension;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{Generics, Ident, ItemStruct, Member, Type};

use crate::FieldsRefExt;

#[extension(pub trait ImplBundle)]
impl ItemStruct {
    fn impl_bundle(&self) -> TokenStream2 {
        let Self {
            ident,
            generics,
            fields,
            ..
        } = self;

        let bundle_impl = fields.tys_ref().zip(fields.members()).fold(
            self.impl_state_bundle(),
            |mut tokens, (ty, member)| {
                impl_field_as_member(ident, generics, ty, member).to_tokens(&mut tokens);
                tokens
            },
        );

        quote! {
            #[expect(unused_imports)]
            const _ : () = {
                use kaleid::state::bundle::*;
                use kaleid::state::KFState;
                use nalgebra::DimNameSum;
                use nalgebra::DimName;
                use nalgebra::Const;
                #bundle_impl
            }
        }
    }

    fn impl_state_bundle(&self) -> TokenStream2 {
        let Self {
            ident,
            generics,
            fields,
            ..
        } = self;

        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        let fields_len = fields.len();
        let first_field_element = fields.tys_ref().next().map(|ty| {
            quote! {
                <#ty as KFState>::Element
            }
        });
        let dims = fields.tys_ref().map(|ty| quote! (<#ty as KFState>::Dim));
        let dim_sum = dims
            .clone()
            .reduce(|dim1, dim2| quote! (DimNameSum<#dim1, #dim2>));

        quote! {
            impl #impl_generics KFState for #ident #ty_generics #where_clause {
                type Element = #first_field_element;
                type Dim = #dim_sum;

                fn gain(&mut self, kalman_gain: ImplVector!(Self::Element, Self::Dim)) {
                    todo!()
                }
            }

            impl #impl_generics StateBundle for #ident #ty_generics #where_clause {
                const DIMS: &'static [usize] = &[
                    #(<#dims as DimName>::DIM),*
                ];
            }

        }
    }
}

fn impl_field_as_member(
    struct_ident: &Ident,
    struct_generics: &Generics,
    field_ty: &Type,
    field_member: Member,
) -> TokenStream2 {
    let (impl_generics, ty_generics, where_clause) = struct_generics.split_for_impl();
    quote! {
        impl #impl_generics AsRef<#field_ty> for #struct_ident #ty_generics #where_clause {
            fn as_ref(&self) -> &#field_ty {
                &self.#field_member
            }
        }

        impl #impl_generics AsMut<#field_ty> for #struct_ident #ty_generics #where_clause {
            fn as_mut(&mut self) -> &mut #field_ty {
                &mut self.#field_member
            }
        }

        impl #impl_generics AsMember<#field_ty> for #struct_ident #ty_generics #where_clause {
            const MEMBER: usize = #field_member;
        }
    }
}

#[test]
fn test_impl() {
    use pretty_assertions::assert_eq;
    use syn::*;

    let input: ItemStruct = parse_quote! {
        struct Foo<T>(
            Bar<T>,
        );
    };
    let expected = quote! {
        impl<T> StateBundle for Foo<T> {
            type Mask = BorrowMask<1usize>;
        }
        impl<T> AsMut< Bar<T> > for Foo<T> {
            fn as_mut(&mut self) -> &mut Bar<T> {
                &mut self.0
            }
        }
        impl<T> AsMember< Bar<T> > for Foo<T> {
            const MEMBER: usize = 0;
        }
    }
    .to_string();

    let output = input.impl_bundle().to_string();

    assert_eq!(output, expected);
}
