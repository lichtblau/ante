use std::sync::Arc;

use inc_complete::DbGet;

use crate::{
    incremental::{GetItem, TypeCheck},
    iterator_extensions::mapvec,
    parser::{
        cst::{self, Name, TopLevelItemKind},
        desugar_context::DesugarContext,
        ids::{NameId, TopLevelId},
    },
    type_inference::types::{Type, TypeBindings},
};

#[derive(Debug)]
pub enum TypeBody {
    Product { type_name: Name, fields: Vec<(Name, Type)> },
    Sum(Vec<(Name, Vec<Type>)>),
}

impl TopLevelId {
    /// Returns the body of this user-defined type (the part after the `=` when declared).
    /// The given [TopLevelId] should refer to a [TypeDefinition] or something which desugars to
    /// one.
    ///
    /// If specified, `arguments` will be used to substitute any generics of the type.
    /// Panics if the arguments are specified and differ in length to the type's generics.
    ///
    /// Note that if `arguments` are not provided, the type will be instantiated and thus
    /// any fields may refer to type type variables that have not been tracked.
    ///
    /// - For a struct: returns each field name & type
    /// - For a union: returns each variant with its name and arguments
    ///
    /// TODO: This function is called somewhat often but is a lot of work to redo each time.
    pub fn type_body<Db>(self, arguments: Option<&[Type]>, compiler: &Db) -> TypeBody
    where
        Db: DbGet<TypeCheck> + DbGet<GetItem>,
    {
        let result = TypeCheck(self).get(compiler);
        let (item, item_context) = GetItem(self).get(compiler);
        let constructor_type = |name| result.get_generalized(name).clone();
        build_type_body(&item.kind, &item_context, arguments, &result.bindings, constructor_type)
    }
}

/// Assemble a type's body from its constructors' types.
///
/// Split out of [`TopLevelId::type_body`] so the checker can reach it while the type's own
/// `TypeCheck` query is still running -- querying it there would be an incremental cycle, and the
/// constructor types it wants are the ones the checker just computed. `constructor_type` supplies
/// each constructor's generalized type; everything else is read from the parsed item.
pub(crate) fn build_type_body(
    kind: &TopLevelItemKind, item_context: &DesugarContext, arguments: Option<&[Type]>, bindings: &TypeBindings,
    constructor_type: impl Fn(NameId) -> Type,
) -> TypeBody {
    let TopLevelItemKind::TypeDefinition(type_definition) = kind else {
        panic!("type_body: passed type_id is not a type!")
    };

    match &type_definition.body {
        cst::TypeDefinitionBody::Struct(fields) => {
            // This'd be easier with an explicit type data field
            let constructor = apply_type_constructor(&constructor_type(type_definition.name), arguments, bindings);
            let field_types = constructor.function_parameter_types();

            assert_eq!(fields.len(), field_types.len());
            let fields = mapvec(fields.iter().zip(field_types), |((field_name, _), typ)| {
                (item_context[*field_name].clone(), typ)
            });

            let type_name = item_context[type_definition.name].clone();
            TypeBody::Product { type_name, fields }
        },
        cst::TypeDefinitionBody::Enum(variants) => {
            let mut variants = mapvec(variants, |(name, _)| {
                let constructor = apply_type_constructor(&constructor_type(*name), arguments, bindings);
                let fields: Vec<_> = constructor.function_parameter_types().collect();
                (item_context[*name].clone(), fields)
            });
            if variants.len() == 1 {
                let (type_name, fields) = variants.pop().unwrap();
                let fields = mapvec(fields.into_iter().enumerate(), |(i, field)| (Arc::new(i.to_string()), field));

                TypeBody::Product { type_name, fields }
            } else {
                TypeBody::Sum(variants)
            }
        },
        // Type aliases are expanded away wherever they are referenced in name resolution, so `type_body`
        // should never be queried for one. `Error` falls through to the same harmless filler.
        cst::TypeDefinitionBody::Alias(_) | cst::TypeDefinitionBody::Error => {
            let type_name = item_context[type_definition.name].clone();
            TypeBody::Product { type_name, fields: Vec::new() }
        },
    }
}

/// Try to apply the given type to the given type arguments. Note that this assumes there are no
/// bound type variables within `typ`!
///
// This assumes constructor args are in the same order as the type args.
// This should be guaranteed by [TypeChecker::build_constructor_type].
fn apply_type_constructor(typ: &Type, args: Option<&[Type]>, bindings: &TypeBindings) -> Type {
    let expected_generic_count = match typ.follow(bindings) {
        Type::Forall(generics, _) => generics.len(),
        _ => 0,
    };

    let arg_len = args.map_or(0, |args| args.len());
    if arg_len != expected_generic_count {
        // TODO: We should be issuing an error either here or above somewhere
    }

    let no_type_var_bindings = TypeBindings::default();

    match args {
        Some(args) => {
            if args.len() < expected_generic_count {
                let mut new_args = args.to_vec();
                for _ in args.len()..expected_generic_count {
                    new_args.push(Type::ERROR);
                }
                typ.apply_type(&new_args, &no_type_var_bindings)
            } else {
                typ.apply_type(args, &no_type_var_bindings)
            }
        },
        None if expected_generic_count == 0 => typ.clone(),
        None => {
            // TODO: This should be an error in the future
            let Type::Forall(generics, _) = typ.follow(bindings) else { unreachable!() };
            let args = mapvec(generics.iter(), |_| Type::ERROR);
            typ.apply_type(&args, &no_type_var_bindings)
        },
    }
}
