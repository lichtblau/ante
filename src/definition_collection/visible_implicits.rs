use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    incremental::{
        AutoDrop, DbHandle, ExportedTypes, GetItem, GetItemRaw, Resolve, VisibleDefinitions, VisibleImplicits,
        VisibleTypes,
    },
    name_resolution::{Origin, namespace::SourceFileId},
    parser::{
        cst::{Name, TopLevelItemKind},
        ids::{NameId, TopLevelId, TopLevelName},
    },
    type_inference::{
        get_type::{get_partial_type, try_get_generalized_type},
        types::{PrimitiveType, Type},
    },
};

/// Maps each ability to its impls. The maps inside are split up for performance so that
/// impl search can search through fewer items.
///
/// E.g. an impl `foo` for `Print (Vec t)` will be stored in:
///   `known_ability_to_impls: [Print -> { type_to_impls: [Vec -> foo] }]`
///
/// Generic types makes this a bit more complex. An impl for a generic type like
/// `bar: Print t` will be stored as:
///   `known_ability_to_impls: [Print -> { generic_type_impls: [bar] }]`
///
/// Similarly, a fully-generic impl `baz: a` will not be in a known ability:
///   `unknown_ability_to_impls: [baz]`
///
/// When searching for a given impl for `Ability Arg`, we must search the `type_to_impls`
/// for that ability & arg pair, the `generic_type_impls` for the ability, and the
/// `unknown_ability_to_impls` which may be impls for any ability.
#[derive(Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Implicits {
    /// Maps each implicit, e.g. `Add` to its impls.
    known_ability_to_impls: BTreeMap<TopLevelId, ImplicitImpls>,

    /// Maps any implicit that does not have a known type to its impls.
    /// We have to check these for every impl which makes these much more expensive.
    unknown_ability_to_impls: Vec<Implicit>,

    /// User-written `Drop` impls whose target is a `shared` type, keyed by that type.
    ///
    /// These are held out of the maps above: they are not the witness for `Drop <SharedType>`.
    /// A shared handle's death is a release, and only the release that reaches zero tears the
    /// pointee down -- so if implicit search resolved `Drop Node` to the user's impl, every dying
    /// alias would run it and the refcount would mean nothing. The witness stays the synthesized
    /// `release_T` wrapper ([`register_shared_drop_impls`]); the user's impl is instead called by
    /// the release glue at count zero, which is the one point where "this value is dying" is true.
    /// The glue finds it here, by name.
    shared_drop_impls: BTreeMap<TopLevelId, Implicit>,
}

type Implicit = (Name, TopLevelName);

#[derive(Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
struct ImplicitImpls {
    /// Maps the first argument of this impl to the set of impls that may
    /// apply to that argument.
    type_to_impls: BTreeMap<TypeKey, Vec<Implicit>>,

    /// Any impls for generic types. These will need to be checked against
    /// all argument types even if an impl in `type_to_impls` matches.
    generic_type_impls: Vec<Implicit>,
}

/// Key each type by its variant for faster mappings. This lets us retrieve
/// a much smaller set of implicits to search through for each type.
///
/// Type applications are stored as their constructor's type key instead.
///
/// Generic types return `None`, we need to sort them separately since they should
/// be checked against every argument type.
///
/// TODO: [Origin] makes this enum too large
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum TypeKey {
    Primitive(PrimitiveType),
    UserDefined(Origin),
    Function,
    Tuple,
    Effects,
}

impl TypeKey {
    /// Converts a type (with no bound type variables) to a [TypeKey].
    /// This returns [None] for generics & type variables so they can be
    /// sorted into another map to be checked against any other type.
    fn from_type(typ: &Type) -> Option<TypeKey> {
        Some(match typ {
            Type::Primitive(primitive) => TypeKey::Primitive(*primitive),
            Type::Generic(_) | Type::Variable(_) => return None,
            Type::Function(_) => TypeKey::Function,
            Type::Application(constructor, _) => return TypeKey::from_type(constructor),
            Type::UserDefined(origin) => TypeKey::UserDefined(*origin),
            // TODO: Is this correct?
            Type::Forall(_, typ) => return TypeKey::from_type(typ),
            Type::Tuple(_) => TypeKey::Tuple,
            Type::U32(_) => return None,
        })
    }
}

/// Returns any global implicits visible to the given item in the context.
/// This will always be a subset of all VisibleDefinitions to the same item.
pub fn visible_implicits_impl(context: &VisibleImplicits, db: &DbHandle) -> Arc<Implicits> {
    let definitions = VisibleDefinitions(context.0).get(db);
    let mut implicits = Implicits::default();
    let drop_ability = drop_ability_id(db);

    for (name, top_level_name) in definitions.definitions.iter() {
        let (item, item_context) = GetItem(top_level_name.top_level_item).get(db);
        let TopLevelItemKind::Definition(definition) = &item.kind else { continue };
        if !definition.implicit {
            continue;
        }

        let resolution = Resolve(top_level_name.top_level_item).get(db);

        // A top-level implicit whose type cannot be derived from its annotation or RHS shape is
        // reported as a missing annotation error and must not contribute to global implicit resolution.
        // Otherwise, we get cascading errors for every implicit search
        if try_get_generalized_type(definition, &item_context, &resolution, db).is_none() {
            continue;
        }

        let typ = get_partial_type(definition, &item_context, &resolution, db, &mut 0);

        // A user `Drop` impl on a shared type is held aside for the release glue rather than
        // registered as the witness -- see [`Implicits::shared_drop_impls`].
        if let Some(drop_ability) = drop_ability
            && let Some(shared_type) = shared_drop_impl_target(&typ, drop_ability, db)
        {
            implicits.shared_drop_impls.insert(shared_type, (name.clone(), *top_level_name));
            continue;
        }

        let mut inserted = false;

        if let Some((ability_id, arg_key)) = get_ability_id_and_first_argument(&typ, true) {
            // Fast path: ability and argument types are known
            let impls = implicits.known_ability_to_impls.entry(ability_id).or_default();

            match arg_key {
                // If this implicit is a function, register it as a candidate for both functions
                // and for its return type
                KeyKind::Function(return_key) => {
                    impls.type_to_impls.entry(TypeKey::Function).or_default().push((name.clone(), *top_level_name));
                    impls.type_to_impls.entry(return_key).or_default().push((name.clone(), *top_level_name));
                },
                KeyKind::Key(key) => {
                    impls.type_to_impls.entry(key).or_default().push((name.clone(), *top_level_name));
                },
                KeyKind::GenericOrUnknown => {
                    impls.generic_type_impls.push((name.clone(), *top_level_name));
                },
            }

            inserted = true;
        }

        if !inserted {
            implicits.unknown_ability_to_impls.push((name.clone(), *top_level_name));
        }
    }

    register_shared_drop_impls(&mut implicits, context.0, db);
    Arc::new(implicits)
}

/// Register the synthesized `Drop` impl of every `shared` type visible here.
///
/// A shared type's teardown is its `release_T` -- decrement, and at zero tear the pointee down and
/// free the block. Nothing expressed that as a `Drop` impl, so a shared handle satisfied no
/// `{Drop t}` constraint, and a container of handles could not release its elements: `Vec`'s own
/// impl needs `Drop t` to drop each one. So a `Vec Node` field of a shared pointee freed its buffer
/// and stranded every subtree hanging off it.
///
/// The witness is registered wherever the type is visible rather than written into a source file,
/// because the type's own module cannot be assumed to import `Drop` and no other module has the
/// right to claim the impl. It is keyed and searched like any other impl from here on; only its
/// type ([`TypeChecker::drop_impl_generalized_type`]) and its body (the MIR builder) are
/// synthesized.
///
/// Gated on `--auto-drop`, like the rest of the Drop-semantics package: with the flag off, a
/// shared type exposes no `Drop` at all and implicit search sees exactly what it saw before.
fn register_shared_drop_impls(implicits: &mut Implicits, file: SourceFileId, db: &DbHandle) {
    let Some(drop_ability) = drop_ability_id(db) else { return };

    for (type_name, type_id) in VisibleTypes(file).get(db).iter() {
        let (item, _) = GetItemRaw(type_id.top_level_item).get(db);
        let TopLevelItemKind::TypeDefinition(definition) = &item.kind else { continue };
        if !definition.shared {
            continue;
        }
        // The key a search for `Drop <SharedType>` computes: the impl's argument is the type
        // itself, and `TypeKey::from_type` reduces an application to its constructor, so a
        // generic shared type keys the same way.
        let key = TypeKey::UserDefined(Origin::TopLevelDefinition(*type_id));
        let witness = TopLevelName::new(type_id.top_level_item, NameId::DROP_IMPL);
        let name = Arc::new(format!("drop_{type_name}"));

        let impls = implicits.known_ability_to_impls.entry(drop_ability).or_default();
        impls.type_to_impls.entry(key).or_default().push((name, witness));
    }
}

/// The Prelude's `Drop` ability, or `None` when the Drop-semantics package is off. Every caller
/// here is part of that package, so `--no-auto-drop` makes them all no-ops together: a shared type
/// exposes no `Drop`, and a user impl on one registers exactly as it did before.
fn drop_ability_id(db: &DbHandle) -> Option<TopLevelId> {
    if !AutoDrop.get(db) {
        return None;
    }
    let prelude_types = ExportedTypes(SourceFileId::prelude()).get(db);
    Some(prelude_types.get(&Arc::new("Drop".to_string()))?.top_level_item)
}

/// The `shared` type a user-written impl implements `Drop` for, if that is what this impl is.
/// A constrained impl (`impl drop_box {Drop t}: Drop (Box t)`) is a function returning the ability
/// type, so its return type is the one that answers.
fn shared_drop_impl_target(typ: &Type, drop_ability: TopLevelId, db: &DbHandle) -> Option<TopLevelId> {
    let (ability_id, arg_key) = get_ability_id_and_first_argument(typ, true)?;
    if ability_id != drop_ability {
        return None;
    }
    let (KeyKind::Key(key) | KeyKind::Function(key)) = arg_key else { return None };
    let TypeKey::UserDefined(Origin::TopLevelDefinition(type_name)) = key else { return None };

    let (item, _) = GetItemRaw(type_name.top_level_item).get(db);
    let TopLevelItemKind::TypeDefinition(definition) = &item.kind else { return None };
    definition.shared.then_some(type_name.top_level_item)
}

enum KeyKind {
    Key(TypeKey),
    /// We can call implicit functions to get their return types so function-typed implicits
    /// get keyed both as a Function and as their return type
    Function(TypeKey),
    /// Generic implicits get put in a separate map for generic types
    GenericOrUnknown,
}

/// If `do_function_check` is true, we allow calling the function for its return type
/// to see if that return type is an ability with arguments. This should be true when
/// registering an implicit to register that it can be called, but not for looking for
/// an implicit in scope.
fn get_ability_id_and_first_argument(typ: &Type, do_function_check: bool) -> Option<(TopLevelId, KeyKind)> {
    match typ {
        Type::Application(constructor, args) => {
            let Some(Origin::TopLevelDefinition(ability_name)) = constructor.as_user_defined() else {
                return None;
            };

            let key = match TypeKey::from_type(&args[0]) {
                Some(key) => KeyKind::Key(key),
                None => KeyKind::GenericOrUnknown,
            };
            Some((ability_name.top_level_item, key))
        },
        Type::Function(function) if do_function_check => {
            let (id, key) = get_ability_id_and_first_argument(&function.return_type, do_function_check)?;
            let key = match key {
                KeyKind::Key(key) => KeyKind::Function(key),
                other => other,
            };
            Some((id, key))
        },
        _ => None,
    }
}

impl Implicits {
    /// The user-written `Drop` impl for a `shared` type, which the release glue calls at count zero.
    /// See [`Implicits::shared_drop_impls`] for why it is not reachable through implicit search.
    pub fn shared_drop_impl(&self, shared_type: TopLevelId) -> Option<TopLevelName> {
        self.shared_drop_impls.get(&shared_type).map(|(_, name)| *name)
    }

    /// Apply `f` to only the implicits that may possibly match the given type
    ///
    /// If there is only one type that matches `target_type` exactly, this is not guaranteed
    /// to iterate over only that type - we may iterate over possibly more items which could
    /// match, but we should never miss a matching definition.
    ///
    /// This is intended as an optimization compared to matching every implicit implicit
    /// in scope without performing any kind of filtering first.
    ///
    /// Return `true` to end the loop early
    pub fn iter_possibly_matching_impls(&self, target_type: &Type, mut f: impl FnMut(&Name, &TopLevelName) -> bool) {
        // Not being able to early-return from a closure means this is simpler as a macro
        macro_rules! apply_f_to_candidates {
            ($f: expr, $candidates: expr) => {{
                for candidate in $candidates {
                    if f(&candidate.0, &candidate.1) {
                        return;
                    }
                }
            }};
        }

        match get_ability_id_and_first_argument(target_type, false) {
            Some((ability_id, KeyKind::Key(argument))) => {
                // Fast case: need to iterate over:
                // 1. (ability match, argument match)
                // 2. (ability match, generic argument)
                if let Some(implicits) = self.known_ability_to_impls.get(&ability_id) {
                    if let Some(candidates) = implicits.type_to_impls.get(&argument) {
                        apply_f_to_candidates!(f, candidates);
                    }
                    apply_f_to_candidates!(f, &implicits.generic_type_impls);
                }
            },
            Some((ability_id, KeyKind::GenericOrUnknown)) => {
                // Unknown argument, need to iterate over every impl of the matching ability
                if let Some(implicits) = self.known_ability_to_impls.get(&ability_id) {
                    for candidates in implicits.type_to_impls.values() {
                        apply_f_to_candidates!(f, candidates);
                    }
                    apply_f_to_candidates!(f, &implicits.generic_type_impls);
                }
            },
            Some((_, KeyKind::Function(_))) => unreachable!("This variant is only used when registering implicits"),
            None => {
                // Unknown ability, need to iterate over everything
                for implicits in self.known_ability_to_impls.values() {
                    for candidates in implicits.type_to_impls.values() {
                        apply_f_to_candidates!(f, candidates);
                    }
                    apply_f_to_candidates!(f, &implicits.generic_type_impls);
                }
            },
        }
        // Finally, for any target type we need to consider all of the impls for unknown abilities
        apply_f_to_candidates!(f, &self.unknown_ability_to_impls);
    }

    /// Return true if there are <= 1 implicits for this type
    pub fn at_most_1_candidate(&self, target_type: &Type) -> bool {
        let mut count = self.unknown_ability_to_impls.len();

        match get_ability_id_and_first_argument(target_type, false) {
            Some((ability_id, KeyKind::Key(argument))) => {
                // Fast case: need to iterate over:
                // 1. (ability match, argument match)
                // 2. (ability match, generic argument)
                if let Some(implicits) = self.known_ability_to_impls.get(&ability_id) {
                    if let Some(candidates) = implicits.type_to_impls.get(&argument) {
                        count += candidates.len();
                    }
                    count += implicits.generic_type_impls.len();
                }
            },
            Some((ability_id, KeyKind::GenericOrUnknown)) => {
                // Unknown argument, need to iterate over every impl of the matching ability
                if let Some(implicits) = self.known_ability_to_impls.get(&ability_id) {
                    for candidates in implicits.type_to_impls.values() {
                        count += candidates.len();
                        if count > 1 {
                            return false;
                        }
                    }
                    count += implicits.generic_type_impls.len();
                }
            },
            Some((_, KeyKind::Function(_))) => unreachable!("This variant is only used when registering implicits"),
            None => {
                // Unknown ability, need to iterate over everything
                for implicits in self.known_ability_to_impls.values() {
                    for candidates in implicits.type_to_impls.values() {
                        count += candidates.len();
                        if count > 1 {
                            return false;
                        }
                    }
                    count += implicits.generic_type_impls.len();
                    if count > 1 {
                        return false;
                    }
                }
            },
        }
        count <= 1
    }
}
