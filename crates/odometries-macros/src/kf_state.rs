use std::borrow::Cow;

use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    FieldsNamed, Generics, Ident, Result, Type, WherePredicate,
    parse::{Parse, ParseStream},
};

use crate::{state_offset::IntoStateOffsetIter, utils::StructNamed};

pub struct KFState {
    pub ident: Ident,
    pub generics: Generics,
    pub fields: FieldsNamed,
    pub element_predicate: Option<WherePredicate>,
    pub element_ty: Type,
    pub last_ty: Option<Type>,
}

impl Parse for KFState {
    fn parse(input: ParseStream) -> Result<Self> {
        let StructNamed {
            attrs,
            ident,
            generics,
            fields,
            ..
        } = input.parse()?;

        let attr_element = attrs
            .iter()
            .find(|attr| attr.path().is_ident("element"))
            .ok_or_else(|| input.error("expect #[element(T: ...)]"))?;

        let (element_ty, element_predicate) = attr_element
            .parse_args::<WherePredicate>()
            .and_then(|predicate| {
                if let WherePredicate::Type(ty) = &predicate {
                    Ok((ty.bounded_ty.clone(), Some(predicate)))
                } else {
                    Err(input.error("expect type predicate"))
                }
            })
            .or_else(|prev_e| {
                attr_element
                    .parse_args::<Type>()
                    .map(|element_ty| (element_ty, None))
                    .map_err(|e| input.error(format!("{prev_e} and {e}")))
            })?;

        let last_ty = fields.named.last().map(|field| &field.ty).cloned();

        Ok(Self {
            ident,
            generics,
            fields,
            element_predicate,
            element_ty,
            last_ty,
        })
    }
}

impl ToTokens for KFState {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let Self {
            ident,
            generics,
            fields,
            element_ty,
            element_predicate,
            last_ty,
        } = self;
        let fields = &fields.named;

        let generics = if let Some(element_predicate) = element_predicate {
            let mut generics = generics.clone();
            generics
                .make_where_clause()
                .predicates
                .push(element_predicate.clone());
            Cow::Owned(generics)
        } else {
            Cow::Borrowed(generics)
        };
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let struct_ty = quote! { #ident #ty_generics };

        let states_offset_impls = fields
            .iter()
            .map(|field| &field.ty)
            .map(|ty| (None, ty))
            .states_offset_impl(ident, &generics);

        let gain_impls = fields
            .iter()
            .filter_map(|field| Some((field.ident.as_ref()?, &field.ty)))
            .map(|(ident, ty)| {
                quote! {
                    self.#ident.gain(rhs.rows_generic(
                        kaleid::state::StateBegin::<Self, #ty>::DIM,
                        <#ty as kaleid::KFState>::Dim::name()
                    ));
                }
            });

        let kf_state_impl = last_ty.as_ref().map(|last_ty| {
            let dim_ty = quote! {
                kaleid::state::StateEnd<#struct_ty, #last_ty>
            };

            quote! {
                impl #impl_generics kaleid::KFState for #struct_ty
                #where_clause
                {
                    type Element = #element_ty;
                    type Dim = #dim_ty;

                    fn gain(
                        &mut self,
                        rhs: nalgebra::Vector<
                            Self::Element,
                            Self::Dim,
                            impl nalgebra::Storage<Self::Element, Self::Dim>
                        >
                    ) {
                        use nalgebra::DimName;
                        #(#gain_impls)*
                    }
                }

            }
        });

        tokens.extend(quote! {
            #(#states_offset_impls)*
            #kf_state_impl
        });
    }
}
