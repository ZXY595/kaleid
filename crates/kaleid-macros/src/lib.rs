use extension_traits::extension;
use itertools::Itertools;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Fields, FieldsUnnamed, GenericArgument, Ident, Item, ItemStruct, Macro, PathArguments, Token,
    Type, TypePath, Visibility, parse::Parser, punctuated::Punctuated,
};
mod bundle;
use bundle::ImplBundle;

#[proc_macro_attribute]
pub fn bundle(arg: TokenStream, ts: TokenStream) -> TokenStream {
    let arg = syn::parse_macro_input!(arg as Option<Ident>);
    let mut item = syn::parse_macro_input!(ts as ItemStruct);

    if let Some(ts) = item.expand_inner_macro() {
        return ts.into();
    };
    item.unique_struct_field();

    let export = arg
        .is_some_and(|ident| ident == "export")
        .then(|| item.export().unwrap_or_compile_error());

    let bundle_impl = item.impl_bundle();
    quote! {
        #item
        #export
        #bundle_impl
    }
    .into()
}

#[proc_macro_attribute]
pub fn extend(arg: TokenStream, ts: TokenStream) -> TokenStream {
    let tys = syn::parse_macro_input!(arg with Punctuated::<Type, Token![,]>::parse_terminated);
    let mut item = syn::parse_macro_input!(ts as Item);
    if let Err(e) = item.extend_tys(tys.into_iter()) {
        return e.to_compile_error().into();
    };
    item.into_token_stream().into()
}

#[extension(trait ItemExt)]
impl Item {
    fn extend_tys(&mut self, tys: impl Iterator<Item = Type>) -> Result<(), syn::Error> {
        match self {
            Self::Macro(item_macro) => {
                let macro_inner = std::mem::take(&mut item_macro.mac.tokens);
                let mut macro_inner_item: Item = syn::parse2(macro_inner)?;
                macro_inner_item.extend_tys(tys)?;
                item_macro.mac.tokens = macro_inner_item.into_token_stream();
                Ok(())
            }
            Self::Struct(item_struct) => {
                item_struct.fields.extend_tys(tys);
                Ok(())
            }
            _ => Err(syn::Error::new_spanned(
                self,
                "expected a macro or a struct",
            )),
        }
    }
}

#[extension(trait FieldsExt)]
impl Fields {
    fn tys(self) -> impl Iterator<Item = Type> {
        self.into_iter().map(|f| f.ty)
    }
    fn from_tys(tys: impl Iterator<Item = Type>) -> Self {
        let fields: FieldsUnnamed = syn::parse_quote!((#(#tys),*));
        Self::Unnamed(fields)
    }
    fn extend_tys(&mut self, tys: impl Iterator<Item = Type>) {
        let fields_tys = self.tys_ref();
        let fields: FieldsUnnamed = syn::parse_quote!((#(#fields_tys,)* #(#tys),*));
        *self = Self::Unnamed(fields);
    }
}

#[extension(trait FieldsRefExt)]
impl<'a> Fields {
    fn tys_ref(&'a self) -> impl Iterator<Item = &'a Type> + Clone {
        self.iter().map(|f| &f.ty)
    }
    fn from_tys_ref(tys: impl Iterator<Item = &'a Type>) -> Self {
        let fields: FieldsUnnamed = syn::parse_quote!((#(#tys),*));
        Self::Unnamed(fields)
    }
}

#[extension(trait DeriveInputExt)]
impl ItemStruct {
    fn expand_inner_macro(&self) -> Option<TokenStream2> {
        let inner_macros = self.fields.iter().filter_map(|f| match &f.ty {
            Type::Macro(m) => Some(&m.mac),
            _ => None,
        });
        inner_macros.clone().next()?;

        let expanded = inner_macros.fold(
            quote!(#[kaleid::bundle]).chain(self.to_token_stream_non_macro()),
            |acc, mac| {
                let Macro {
                    path,
                    bang_token,
                    tokens,
                    ..
                } = mac;
                Punctuated::<Ident, Token![,]>::parse_terminated
                    .parse2(tokens.clone())
                    .map(|generics| {
                        quote! {
                            #path #bang_token {
                                #[generics(#generics)]
                                #acc
                            }
                        }
                    })
                    .unwrap_or_compile_error()
            },
        );
        Some(expanded)
    }

    fn to_token_stream_non_macro(&self) -> TokenStream2 {
        let Self {
            attrs,
            vis,
            struct_token,
            ident,
            generics,
            fields,
            semi_token,
        } = self;

        let tys = fields.iter().filter_map(|f| match &f.ty {
            Type::Macro(_) => None,
            _ => Some(&f.ty),
        });
        quote! {
            #(#attrs)*
            #vis #struct_token #ident #generics
            (#(#tys),*) #semi_token
        }
    }

    fn unique_struct_field(&mut self) {
        let tys = self.fields.tys_ref().unique();
        self.fields = Fields::from_tys_ref(tys);
        self.semi_token = Some(Default::default());
    }

    fn export(&self) -> Result<TokenStream2, syn::Error> {
        let Self {
            vis,
            ident,
            fields,
            generics,
            ..
        } = self;
        let ident = format_ident!("{ident}Bundle");
        let (macro_export, reexport) = match vis {
            Visibility::Public(public) => {
                (quote! { #[macro_export] }, quote! { #public use #ident; })
            }
            Visibility::Restricted(vis_restricted) => {
                (TokenStream2::new(), quote! { #vis_restricted use #ident; })
            }
            Visibility::Inherited => (TokenStream2::new(), quote! { use #ident; }),
        };
        let tys = fields
            .iter()
            .map(|f| match &f.ty {
                syn::Type::Path(type_path) => type_path
                    .is_crate_relative()
                    .then(|| type_path.to_macro_args_generics())
                    .ok_or_else(|| {
                        syn::Error::new_spanned(type_path, "expected a crate-relative path")
                    }),
                _ => Err(syn::Error::new_spanned(&f.ty, "not a path type")),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let generic_args = generics.type_params().map(|p| &p.ident);

        Ok(quote! {
            #macro_export
            macro_rules! #ident {
                (#[generics(#($#generic_args:ident),*)]
                 $($rest:tt)*) => {
                    #[kaleid::extend(#($#tys),*)]
                    $($rest)*
                }
            }
            #reexport
        })
    }
}

#[extension(trait TypePathExt)]
impl TypePath {
    fn is_crate_relative(&self) -> bool {
        self.path
            .segments
            .first()
            .is_some_and(|seg| seg.ident == "crate")
    }

    fn to_macro_args_generics(&self) -> Self {
        let mut type_path = self.clone();
        type_path.map_generics_to_macro_args();
        type_path
    }

    fn map_generics_to_macro_args(&mut self) {
        self.path
            .segments
            .iter_mut()
            .filter_map(|seg| match &mut seg.arguments {
                PathArguments::AngleBracketed(args) => Some(args),
                _ => None,
            })
            .flat_map(|angled_args| angled_args.args.iter_mut())
            .filter_map(|args| match args {
                GenericArgument::Type(ty) => Some(ty),
                _ => None,
            })
            .for_each(|ty| {
                let Type::Path(type_path) = &ty else {
                    return;
                };
                *ty = Type::Verbatim(quote!( $#type_path ));
            });
    }
}

#[extension(trait TokenStream2Ext)]
impl TokenStream2 {
    fn chain(mut self, other: TokenStream2) -> TokenStream2 {
        other.to_tokens(&mut self);
        self
    }
}

#[extension(trait TryTokenStream2Ext)]
impl Result<TokenStream2, syn::Error> {
    fn unwrap_or_compile_error(self) -> TokenStream2 {
        self.unwrap_or_else(|e| e.into_compile_error())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use quote::quote;
    use syn::*;

    #[test]
    fn test_expand_inner_macro() {
        let input: ItemStruct = parse_quote! {
            #[derive(Debug)]
            pub struct A<T>(some_bundle!(), T, u8, other_bundle!());
        };
        let expected = quote! {
            other_bundle! {
                some_bundle! {
                    #[kaleid::bundle]
                    #[derive(Debug)]
                    pub struct A<T>(T, u8);
                }
            }
        }
        .to_string();

        let output = input.expand_inner_macro().unwrap().to_string();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_extend_tys() {
        let mut input: Item = parse_quote! {
            #[derive(Debug)]
            pub struct A<T>(T);
        };
        let expected = quote! {
            #[derive(Debug)]
            pub struct A<T>(T, u32, B<T>);
        }
        .to_string();

        input
            .extend_tys([parse_quote! { u32 }, parse_quote! { B<T> }].into_iter())
            .unwrap();
        let output = input.into_token_stream().to_string();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_unique_unnamed() {
        let mut input: ItemStruct = parse_quote! {
            #[derive(Debug)]
            pub struct A<T>(u32, T, u32);
        };
        let expected = quote! {
            #[derive(Debug)]
            pub struct A<T>(u32, T);
        }
        .to_string();

        input.unique_struct_field();
        let output = input.into_token_stream().to_string();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_unique_named() {
        let mut input: ItemStruct = parse_quote! {
            #[derive(Debug)]
            pub struct A {
                a: u32,
                b: u32,
            }
        };
        let expected = quote! {
            #[derive(Debug)]
            pub struct A(u32);
        }
        .to_string();

        input.unique_struct_field();
        let output = input.into_token_stream().to_string();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_export() {
        let input: ItemStruct = parse_quote! {
            #[derive(Debug)]
            pub struct A<T> where T: Test {
                a: crate::Vector3<T>,
                b: crate::Vector3<T>,
                c: crate::Vector2<T>,
            }
        };
        let expected = quote! {
            #[macro_export]
            macro_rules! A {
                (#[generics($T:ident)]
                 $($rest:tt)*) => {
                    #[kaleid::extend(
                        $crate::Vector3<$T>,
                        $crate::Vector3<$T>,
                        $crate::Vector2<$T>
                    )]
                    $($rest)*
                }
            }
            pub use A;
        }
        .to_string();

        let output = input
            .export()
            .unwrap_or_compile_error()
            .into_token_stream()
            .to_string();

        assert_eq!(output, expected);
    }
}
