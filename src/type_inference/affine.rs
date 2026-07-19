use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    diagnostics::{Diagnostic, Location, RepeatedContext},
    incremental::{ExportedTypes, GetItemRaw, VisibleImplicits},
    name_resolution::Origin,
    parser::{
        cst::{Expr, TopLevelItemKind},
        ids::{ExprId, NameId, TopLevelId, TopLevelName},
    },
    type_inference::{Locateable, TypeChecker, types::Type},
};

use super::fresh_expr::ExtendedTopLevelContext;

use crate::name_resolution::namespace::SourceFileId;

/// A path that can be moved: either a variable or a chain of field accesses.
/// For example, `x` or `x.one.two`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum MovePath {
    Variable(NameId),
    Field(Box<MovePath>, String),
}

impl MovePath {
    /// Project a field out of a parent place, e.g. `whole` + `"a"` becomes `whole.a`.
    pub(super) fn field(parent: MovePath, field: String) -> MovePath {
        MovePath::Field(Box::new(parent), field)
    }

    /// Check if `self` is a proper descendant of `ancestor` (but is not itself the ancestor).
    /// E.g. `x.a.b` is a descendant of `x.a` and `x`, but not of `x.a.b`.
    fn is_descendant_of(&self, ancestor: &MovePath) -> bool {
        match self {
            _ if self == ancestor => false,
            MovePath::Field(parent, _) => parent.as_ref() == ancestor || parent.is_descendant_of(ancestor),
            MovePath::Variable(_) => false,
        }
    }

    /// If `self` is a proper descendant of `prefix`, return a copy with the `prefix` head
    /// replaced by `new_root` (e.g. `x.a.b` with prefix `x.a` and new root `p` becomes `p.b`).
    /// Returns `None` when `self` is `prefix` itself or is not under it. Used to carry the
    /// partial moves recorded under an enum payload place (`s.Ok#0`, which has no expression
    /// form) onto the fresh binding a sum drop projects that payload into.
    fn reroot(&self, prefix: &MovePath, new_root: &MovePath) -> Option<MovePath> {
        match self {
            _ if self == prefix => None,
            MovePath::Field(parent, field) => {
                if parent.as_ref() == prefix {
                    Some(MovePath::field(new_root.clone(), field.clone()))
                } else {
                    Some(MovePath::field(parent.reroot(prefix, new_root)?, field.clone()))
                }
            },
            MovePath::Variable(_) => None,
        }
    }

    /// Return the root variable name of this path.
    /// E.g. for `x.one.two`, returns the NameId of `x`.
    pub(super) fn root_variable(&self) -> NameId {
        match self {
            MovePath::Variable(name) => *name,
            MovePath::Field(parent, _) => parent.root_variable(),
        }
    }

    /// Build a display name for error messages, e.g. `"c.one.two"`.
    pub(super) fn display_name(&self, context: &ExtendedTopLevelContext) -> String {
        match self {
            MovePath::Variable(name_id) => context[*name_id].to_string(),
            MovePath::Field(parent, field) => {
                format!("{}.{}", parent.display_name(context), field)
            },
        }
    }
}

/// Tracks which paths have been moved in the current scope.
/// Used for affine type checking: non-Copy values may only be used once.
#[derive(Clone, Default)]
pub(super) struct MoveTracker {
    moved: FxHashMap<MovePath, Location>,
    errored: FxHashSet<MovePath>,
}

/// A snapshot of a [`MovePath`]'s move record, taken before an expression is inferred so
/// the auto-ref coercion can roll the place back to its pre-inference state.
pub(super) struct SavedMove {
    pub(super) path: MovePath,
    pub(super) location: Option<Location>,
}

impl MoveTracker {
    /// Record that a path has been moved at the given location.
    pub(super) fn record_move(&mut self, path: MovePath, location: Location) {
        self.moved.insert(path, location);
    }

    /// Clear any move record for `path` and its descendants. Called when `path` is being reassigned.
    pub(super) fn clear_moves(&mut self, path: &MovePath) {
        self.moved.remove(path);
        self.moved.retain(|p, _| !p.is_descendant_of(path));
        self.errored.remove(path);
        self.errored.retain(|p| !p.is_descendant_of(path));
    }

    /// Snapshot the move record for `path` so it can be restored after an auto-ref coercion.
    pub(super) fn save_move(&self, path: &MovePath) -> Option<Location> {
        self.moved.get(path).cloned()
    }

    /// Restore a previously saved move record, dropping this expression's own contribution.
    pub(super) fn restore_move(&mut self, path: &MovePath, saved: Option<Location>) {
        match saved {
            Some(location) => self.moved.insert(path.clone(), location),
            None => self.moved.remove(path),
        };
    }

    /// Check if this path or any ancestor is already moved.
    /// Returns the location of the move if found.
    pub(super) fn is_moved(&self, path: &MovePath) -> Option<&Location> {
        if let Some(loc) = self.moved.get(path) {
            return Some(loc);
        }
        match path {
            MovePath::Field(parent, _) => self.is_moved(parent),
            MovePath::Variable(_) => None,
        }
    }

    /// Check if any child (descendant) of this path has been moved.
    /// Returns the first one found, if any.
    pub(super) fn has_child_moved(&self, path: &MovePath) -> Option<(&MovePath, &Location)> {
        self.moved.iter().find(|(moved_path, _)| moved_path.is_descendant_of(path))
    }

    /// Build a fresh tracker holding this tracker's moves that lie strictly under `prefix`,
    /// with their `prefix` head replaced by `new_root`. A residual sum drop projects an enum
    /// payload (`s.Ok#0`) into a fresh binding to drop it, so the sub-place moves recorded under
    /// the payload (a nested binding like `d` in `Ok (d, _root)` moved `s.Ok#0.0` out) must be
    /// re-rooted onto that binding for its residual drop to skip the parts already moved away.
    pub(super) fn reroot_descendants(&self, prefix: &MovePath, new_root: &MovePath) -> MoveTracker {
        let mut result = MoveTracker::default();
        for (path, location) in &self.moved {
            if let Some(rerooted) = path.reroot(prefix, new_root) {
                result.moved.insert(rerooted, location.clone());
            }
        }
        result
    }

    /// The root names of whole-local moves (`MovePath::Variable` entries) recorded here.
    /// Used by drop elaboration's branch-edge equalization; field-path (partial) moves are
    /// handled separately by residual drops.
    pub(super) fn whole_moved_locals(&self) -> impl Iterator<Item = NameId> + '_ {
        self.moved.keys().filter_map(|path| match path {
            MovePath::Variable(name) => Some(*name),
            MovePath::Field(..) => None,
        })
    }

    /// Merge into `self` the moves from `other` whose root variable is in `roots`.
    /// Used (under `--auto-drop`) to surface a lambda body's moves of captured outer
    /// variables to the enclosing scope: closures capture by reference, so a moved capture
    /// is gone from the outer scope's perspective and must not be dropped there again.
    /// Over-reporting is safe (a never-run closure's "move" just skips a drop -- a leak),
    /// under-reporting is a double-free.
    pub(super) fn merge_moves_rooted_in(&mut self, other: &MoveTracker, roots: &FxHashSet<NameId>) {
        for (path, location) in &other.moved {
            if roots.contains(&path.root_variable()) && !self.moved.contains_key(path) {
                self.moved.insert(path.clone(), location.clone());
            }
        }
        for path in &other.errored {
            if roots.contains(&path.root_variable()) {
                self.errored.insert(path.clone());
            }
        }
    }

    /// Merge move trackers from multiple branches.
    /// A path is considered moved after the branch if it was moved in the base
    /// OR in ANY branch (since one of the branches will execute).
    pub(super) fn merge_branches(base: &MoveTracker, branches: &[MoveTracker]) -> MoveTracker {
        let mut result = base.clone();
        for branch in branches {
            for (path, loc) in &branch.moved {
                if !result.moved.contains_key(path) {
                    result.moved.insert(path.clone(), loc.clone());
                }
            }
            for path in &branch.errored {
                result.errored.insert(path.clone());
            }
        }
        result
    }
}

impl<'local, 'inner> TypeChecker<'local, 'inner> {
    /// Returns the TopLevelName for the Prelude's `Copy` type, caching it.
    fn get_copy_type_name(&mut self) -> TopLevelName {
        if let Some(name) = self.copy_type_name {
            return name;
        }
        let exported_types = ExportedTypes(SourceFileId::prelude()).get(self.compiler);
        let top_level_name = exported_types.get(&Arc::new("Copy".to_string())).expect("Copy type not found in Prelude");
        self.copy_type_name = Some(*top_level_name);
        *top_level_name
    }

    /// Returns true if the given type implements Copy.
    ///
    /// TODO: Write the actual implicit call to Copy when a copy variable is used.
    pub(super) fn type_is_copy(&mut self, typ: &Type) -> bool {
        let typ = self.follow_type(typ).clone();

        // Fast path: all primitive types are Copy (uniq refs are Type::Applications)
        if matches!(&typ, Type::Primitive(_)) {
            return true;
        }

        // TODO: This isn't always true, but we also can't define the proper Copy impls
        // for functions in the stdlib because we can't manually access a closure's environment
        // and we can't define every copy impl for every possible parameter count.
        if matches!(&typ, Type::Function(_)) {
            return true;
        }

        // Tuple types are Copy if all elements are Copy
        if let Type::Tuple(elems) = &typ {
            return elems.iter().all(|e| self.type_is_copy(e));
        }

        // TODO: Actually require abilities only capture `Copy` types
        if self.is_ability(&typ) {
            return true;
        }

        // `shared` types are pointer-wrapped in MIR and are always Copy.
        if self.is_shared_user_defined(&typ) {
            return true;
        }

        let copy_name = self.get_copy_type_name();

        // Under --auto-drop, bare type variables are handled honestly. The historic search
        // below lets an unbound variable unify with any concrete impl's target (`Copy I8`),
        // silently treating every generic value as Copy: moves unrecorded, drops skipped --
        // and a monomorphization-time double-free once `t = String`. A variable is Copy iff
        // it is an int/float literal variable (it will default to a Copy primitive) or an
        // in-scope `{Copy t}` constraint names exactly this variable (unification is too
        // loose even for local implicits: it would bind an unrelated `{Copy u}`'s variable).
        if self.auto_drop && let Type::Variable(id) = &typ {
            if self.is_literal_variable(*id) {
                return true;
            }
            if self.constraint_in_scope_for_variable(*id, copy_name) {
                return true;
            }
            if self.is_signature_variable(*id) {
                return false;
            }
            // In-flight unification variables (a lambda parameter before its call site
            // unifies it, an unconstrained element type) keep the legacy lenient search
            // below -- treating them as affine mid-flight would reject loop-carried uses
            // of values that end up Copy.
        }

        let copy_constructor = Type::UserDefined(Origin::TopLevelDefinition(copy_name));

        let copy_of_t = Type::Application(Arc::new(copy_constructor), Arc::new(vec![typ.clone()]));

        // Check local implicits in scope
        let local_implicits = self.collect_implicits_in_scope();
        for name in &local_implicits {
            // Head pre-filter: most in-scope implicits are other abilities (`{Drop t}`,
            // `{Cmp t}`); their unification against `Copy _` can only fail, so skip the
            // `follow_all` and the attempt.
            if !self.implicit_could_be_ability(&self.name_types[name], copy_name) {
                continue;
            }
            let name_type = self.name_types[name].follow_all(&self.bindings);
            if self.try_unify(&name_type, &copy_of_t).is_ok() {
                return true;
            }
        }

        // Check global implicits. The search unifies against every candidate impl; for a
        // fully-concrete type its outcome cannot change with later unification and the local
        // implicits were already consulted above, so memoize it per (source file, type). The
        // raw type keys the cache -- see `get_field_types`.
        let Some(item) = self.current_item else { return false };
        // Probe before the concreteness walk: non-concrete keys are never inserted, so
        // their lookups just miss (see `type_needs_no_drop`).
        let probe_key = (item.source_file, typ.clone());
        if let Some(hit) = self.copy_search_cache.get(&probe_key) {
            return *hit;
        }

        let visible_implicits = VisibleImplicits(item.source_file).get(self.compiler);
        let mut found = false;
        visible_implicits.iter_possibly_matching_impls(&copy_of_t, |_name, name_id| {
            let (name_type, _) = self.type_and_bindings_of_top_level_name(name_id);
            if self.try_unify(&name_type, &copy_of_t).is_ok() {
                found = true;
                return true;
            }
            // Also check if it's a function whose return type matches. Under --auto-drop
            // the candidate's own implicit constraints must hold too: `copy_maybe
            // {Copy a}: Copy (Maybe a)` must not make `Maybe NonCopy` Copy, or its
            // payload would never be tracked or dropped. (Without the flag the historic
            // constraint-blind behavior is kept so default checking is unchanged.)
            if let Type::Function(f) = &name_type
                && let Ok(bindings) = self.try_unify(&f.return_type, &copy_of_t)
            {
                let f = f.clone();
                if !self.auto_drop || self.copy_impl_constraints_hold(&f, bindings) {
                    found = true;
                    return true;
                }
            }
            false
        });
        if self.type_is_concrete(&probe_key.1) {
            self.copy_search_cache.insert(probe_key, found);
        }
        found
    }

    /// True if the variable will default to an int/float primitive (both Copy): either it
    /// is itself a literal variable, or some literal variable's binding chain leads to it
    /// (unification may bind the literal variable to another variable, making that one the
    /// representative -- e.g. the parameters of a desugared `loop (len = 0)`).
    pub(super) fn is_literal_variable(&self, id: super::types::TypeVariableId) -> bool {
        if self.integer_literal_vars.contains(&id) || self.float_literal_vars.contains(&id) {
            return true;
        }
        let follows_to_id = |lit: &super::types::TypeVariableId| {
            matches!(Type::Variable(*lit).follow(&self.bindings), Type::Variable(v) if *v == id)
        };
        self.integer_literal_vars.iter().any(follows_to_id) || self.float_literal_vars.iter().any(follows_to_id)
    }

    /// True if the variable is (or is the binding representative of) one of the current
    /// item's signature type variables -- a rigid generic, which gets honest Copy/Drop
    /// treatment under `--auto-drop`.
    pub(super) fn is_signature_variable(&self, id: super::types::TypeVariableId) -> bool {
        if self.signature_type_vars.contains(&id) {
            return true;
        }
        self.signature_type_vars
            .iter()
            .any(|var| matches!(Type::Variable(*var).follow(&self.bindings), Type::Variable(v) if *v == id))
    }

    /// Cheap pre-filter for the local-implicit scans: could this implicit's type possibly be
    /// (or unify with) `<ability> _`? Only an application headed by the ability itself, or
    /// something unification could still bind (an unbound head or a wholly-unbound type),
    /// can. Head-only follows, no allocation -- the scans previously paid a deep
    /// `follow_all` + unification attempt per implicit per query, which dominated profiles.
    /// `true` is the conservative answer (the caller just attempts the match as before).
    pub(super) fn implicit_could_be_ability(&self, typ: &Type, ability: TopLevelName) -> bool {
        match typ.follow(&self.bindings) {
            Type::Application(constructor, _) => match constructor.follow(&self.bindings) {
                Type::UserDefined(Origin::TopLevelDefinition(name)) => *name == ability,
                Type::Variable(_) | Type::Generic(_) => true,
                _ => false,
            },
            Type::Variable(_) | Type::Generic(_) => true,
            Type::Forall(_, inner) => self.implicit_could_be_ability(inner, ability),
            _ => false,
        }
    }

    /// True if a local implicit of shape `<ability> x` -- where `x` follows to exactly the
    /// given rigid/unbound generic type (a named generic or a bare type variable) -- is in
    /// scope. Used where unifying against candidates is too loose: unification would bind
    /// the variable to whatever it is compared with instead of matching it.
    pub(super) fn constraint_in_scope_for_generic(&mut self, target: &Type, ability: TopLevelName) -> bool {
        let target = target.follow(&self.bindings).clone();
        let local_implicits = self.collect_implicits_in_scope();
        for name in &local_implicits {
            let Some(name_type) = self.name_types.get(name) else { continue };
            if !self.implicit_could_be_ability(name_type, ability) {
                continue;
            }
            let name_type = name_type.follow_all(&self.bindings);
            let Type::Application(constructor, args) = &name_type else { continue };
            let matches_ability = matches!(
                constructor.follow(&self.bindings),
                Type::UserDefined(Origin::TopLevelDefinition(name)) if *name == ability
            );
            if matches_ability
                && let Some(arg) = args.first()
                && *arg.follow(&self.bindings) == target
            {
                return true;
            }
        }
        false
    }

    /// [`Self::constraint_in_scope_for_generic`] for a bare type-variable target.
    pub(super) fn constraint_in_scope_for_variable(
        &mut self, var: super::types::TypeVariableId, ability: TopLevelName,
    ) -> bool {
        self.constraint_in_scope_for_generic(&Type::Variable(var), ability)
    }

    /// Check that a candidate Copy impl's implicit `{Copy x}` constraints are satisfiable
    /// under the unification `bindings` produced by matching its return type. Non-Copy
    /// constraints are assumed satisfiable (over-approximation, matching the search proper).
    /// A depth guard bounds constraint-driven recursion; running out means "not Copy" --
    /// the safe direction for drops (the value gets tracked and dropped, not duplicated).
    fn copy_impl_constraints_hold(
        &mut self, function: &crate::type_inference::types::FunctionType, bindings: super::types::TypeBindings,
    ) -> bool {
        if self.copy_check_depth >= 8 {
            return false;
        }
        self.copy_check_depth += 1;
        let mut merged = self.bindings.clone();
        merged.extend(bindings);
        let copy_name = self.get_copy_type_name();
        let mut holds = true;
        for parameter in function.parameters.iter().filter(|parameter| parameter.is_implicit) {
            let constraint = parameter.typ.follow_all(&merged);
            let Type::Application(constructor, args) = &constraint else { continue };
            let is_copy_constraint = matches!(
                constructor.follow(&merged),
                Type::UserDefined(Origin::TopLevelDefinition(name)) if *name == copy_name
            );
            if !is_copy_constraint {
                continue;
            }
            let Some(arg) = args.first() else { continue };
            let arg = arg.follow_all(&merged);
            if !self.type_is_copy(&arg) {
                holds = false;
                break;
            }
        }
        self.copy_check_depth -= 1;
        holds
    }

    fn is_ability(&self, typ: &Type) -> bool {
        match typ.follow(&self.bindings) {
            // Type aliases are expanded away during `from_cst_type`, so no `UserDefined`
            // here can refer to an alias
            Type::Application(constructor, _) => self.is_ability(constructor),
            Type::UserDefined(origin) => match origin {
                Origin::TopLevelDefinition(name) => {
                    if let Some(hit) = self.ability_cache.borrow().get(&name.top_level_item) {
                        return *hit;
                    }
                    let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
                    let is_ability = matches!(&item.kind, TopLevelItemKind::AbilityDefinition(_));
                    self.ability_cache.borrow_mut().insert(name.top_level_item, is_ability);
                    is_ability
                },
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns the `(shared, mutable)` flags if `typ` resolves to a user-defined type definition.
    fn shared_type_flags(&self, typ: &Type) -> Option<(bool, bool)> {
        match typ.follow(&self.bindings) {
            Type::Application(constructor, _) => self.shared_type_flags(constructor),
            Type::UserDefined(Origin::TopLevelDefinition(name)) => {
                if let Some(hit) = self.shared_flags_cache.borrow().get(&name.top_level_item) {
                    return *hit;
                }
                let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
                let flags = match &item.kind {
                    TopLevelItemKind::TypeDefinition(td) => Some((td.shared, td.mutable)),
                    _ => None,
                };
                self.shared_flags_cache.borrow_mut().insert(name.top_level_item, flags);
                flags
            },
            _ => None,
        }
    }

    pub(super) fn is_shared_user_defined(&self, typ: &Type) -> bool {
        matches!(self.shared_type_flags(typ), Some((true, _)))
    }

    /// True when `typ` is a closure whose environment is a bare `Pointer` -- the exact shape the MIR
    /// builder heap-allocates via `AllocShared`. Such closure values are reference-counted like
    /// shared handles: their env pointer carries a refcount header, so a copy retains it and a
    /// death releases it. A `Ptr Unit` dictionary environment (ability method used as a first-class
    /// value -- uniform-representation overhead, not a real capture) follows to `Type::Application`,
    /// not the bare `Pointer` primitive, and is excluded. A slot of this shape filled by a
    /// capture-less value carries a null env pointer, which the null-safe retain/release intrinsics
    /// skip at runtime.
    pub(super) fn is_heap_env_closure(&self, typ: &Type) -> bool {
        match self.follow_type(typ) {
            Type::Function(function) => {
                matches!(function.environment.follow(&self.bindings), Type::Primitive(super::types::PrimitiveType::Pointer))
            },
            _ => false,
        }
    }

    /// If `typ` resolves to a `shared` user-defined type, returns its defining item's id. Used to
    /// name that type's synthesized `release_T` -- the same id both a release call site and the
    /// type's own MIR emission derive the release function name from.
    pub(super) fn shared_type_top_level_id(&self, typ: &Type) -> Option<TopLevelId> {
        match typ.follow(&self.bindings) {
            Type::Application(constructor, _) => self.shared_type_top_level_id(constructor),
            Type::UserDefined(Origin::TopLevelDefinition(name)) => {
                let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
                match &item.kind {
                    TopLevelItemKind::TypeDefinition(td) if td.shared => Some(name.top_level_item),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    pub(super) fn is_shared_mut_user_defined(&self, typ: &Type) -> bool {
        matches!(self.shared_type_flags(typ), Some((true, true)))
    }

    /// Check if using `path` is valid (not already moved or partially moved).
    /// Emits a diagnostic if the path was already moved.
    /// Only emits the first error per path to avoid noisy duplicate diagnostics.
    pub(super) fn check_use_of_move_path(&mut self, path: &MovePath, locator: impl Locateable) {
        if self.move_tracker.errored.contains(path) {
            return;
        }

        // Auto-drop: tentative move records exist for bare generic variables whose
        // Copy-ness is not settled yet (see `infer_path`). While the value's type still
        // reads as Copy, a recorded "move" must not produce use-of-moved errors -- it only
        // informs the drop planner. Once the type is provably non-Copy (a `{Drop t}`-bound
        // signature generic, or a later-bound concrete type), errors fire as usual.
        if self.auto_drop
            && let Some(root_type) = self.name_types.get(&path.root_variable()).cloned()
            && self.type_is_copy(&root_type)
        {
            return;
        }

        // Check if this exact path or an ancestor was moved
        if let Some(moved_loc) = self.move_tracker.is_moved(path) {
            let name = path.display_name(self.current_extended_context());
            let location = locator.locate(self);
            let moved_in = moved_loc.clone();
            self.compiler.accumulate(Diagnostic::UseOfMovedValue { name: name.clone(), location, moved_in });
            self.move_tracker.errored.insert(path.clone());

        // Check if any child was moved (partial move)
        } else if let Some((_child_path, moved_loc)) = self.move_tracker.has_child_moved(path) {
            let name = path.display_name(self.current_extended_context());
            let location = locator.locate(self);
            let moved_in = moved_loc.clone();
            self.compiler.accumulate(Diagnostic::UseOfMovedValue { name, location, moved_in });
            self.move_tracker.errored.insert(path.clone());
        }
    }

    /// Emit errors for any non-Copy outer variables moved during a context whose
    /// body may run more than once (handler branches, `for` bodies, `while`
    /// condition + body). Call this with `self.move_tracker` set to the scope-local
    /// tracker (started empty via `mem::take`); `outer_names` is the set of NameIds
    /// that existed *before* the scope was entered.
    pub(super) fn check_moves_in_repeated_context(
        &mut self, outer_names: &rustc_hash::FxHashSet<NameId>, context: RepeatedContext,
    ) {
        let outer_moves: Vec<(MovePath, Location)> = self
            .move_tracker
            .moved
            .iter()
            .filter(|(path, _)| outer_names.contains(&path.root_variable()))
            .map(|(p, l)| (p.clone(), l.clone()))
            .collect();

        for (path, location) in outer_moves {
            if !self.type_is_copy(&self.name_types[&path.root_variable()].clone()) {
                let name = path.display_name(self.current_extended_context());
                self.compiler.accumulate(Diagnostic::MoveInRepeatedContext { name, context, location });
            }
        }
    }

    /// The place a binding denotes: a recorded sub-place, or its own variable by default.
    pub(super) fn binding_place(&self, name: NameId) -> MovePath {
        self.binding_places.get(&name).cloned().unwrap_or(MovePath::Variable(name))
    }

    /// Try to build a MovePath from an expression by walking through
    /// variable references and member access chains.
    /// Returns None if the expression is not a simple path.
    pub(super) fn try_build_move_path(&self, expr: ExprId) -> Option<MovePath> {
        match &self.current_extended_context()[expr] {
            Expr::Variable(path) => {
                if let Some(Origin::Local(name)) = self.path_origin(*path) {
                    Some(self.binding_place(name))
                } else {
                    None
                }
            },
            Expr::MemberAccess(access) => {
                let parent = self.try_build_move_path(access.object)?;
                Some(MovePath::field(parent, access.member.clone()))
            },
            _ => None,
        }
    }
}
