//! Drop elaboration
//!
//! While the checker traverses an item, a stack of [`DropScope`]s records which owned
//! locals each scope declares. At every scope-exit edge the live [`super::affine::MoveTracker`]
//! state decides what is still owned there, and each obligation is materialized as a
//! synthesized `drop (mut <local>)` call in the [`super::fresh_expr::ExtendedTopLevelContext`],
//! checked through the ordinary inference pipeline so implicit search resolves the `Drop`
//! impl exactly like a user-written drop. The call's `ExprId`s are recorded in side tables
//! (`post_expr_drops` / `pre_exit_drops`) that the MIR builder lowers at the matching exits.
//!
//! Elaboration must run inline during inference (not as a post-pass): implicit scopes --
//! including `{Drop t}` capability parameters -- are popped as each scope ends, so a
//! post-pass could no longer resolve the impls.
//!
//! Interim limitations: partially-moved values are skipped (no residual drops yet), types that are
//! not fully concrete are skipped (generic drops need `{Drop t}` propagation), and types with no
//! visible `Drop` impl are skipped.

use std::sync::Arc;

use crate::{
    diagnostics::Location,
    incremental::{ExportedDefinitions, VisibleImplicits},
    name_resolution::{Origin, namespace::SourceFileId},
    parser::{
        cst::{self, Expr, ReferenceKind},
        ids::{ExprId, NameId, PatternId, TopLevelName},
    },
    type_inference::{TypeChecker, affine::MovePath, errors::TypeErrorKind, types::Type},
};

/// The kind of scope a [`DropScope`] tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DropScopeKind {
    /// A block (`Expr::Sequence`); exits by fallthrough at its last statement.
    Block,
    /// A function body. Owns the parameters, and is the boundary `return` unwinds to.
    Function,
}

/// One entry of the drop-scope stack: the owned-root locals a scope declares,
/// in declaration order (dropped in reverse).
pub(super) struct DropScope {
    pub(super) kind: DropScopeKind,
    pub(super) names: Vec<NameId>,
}

impl TypeChecker<'_, '_> {
    /// True while drop elaboration should observe the traversal: `--auto-drop` is on, we are
    /// not inside the inference of a synthesized drop call itself, and moves are being
    /// recorded. The last condition matters for coercion-wrapper re-checks (`coerce`'s
    /// ReplacedExpr path re-infers the wrapper with moves suppressed): elaborating there
    /// would see the wrapper's params as never-moved and drop values its body forwarded --
    /// a double-free (e.g. the eta-expansion wrapper for `(++)` passed to `foldl`).
    pub(super) fn drop_elaboration_active(&self) -> bool {
        self.auto_drop && !self.synthesizing_drops && !self.suppress_move_record
    }

    pub(super) fn push_drop_scope(&mut self, kind: DropScopeKind) {
        if self.drop_elaboration_active() {
            self.drop_scopes.push(DropScope { kind, names: Vec::new() });
        }
    }

    /// Pop the innermost drop scope and synthesize the ordered drop calls for its
    /// fallthrough edge. `diverges` skips the edge entirely (it is unreachable, and any
    /// early exit recorded its own drops).
    pub(super) fn pop_drop_scope(&mut self, diverges: bool, location: &Location) -> Vec<ExprId> {
        if !self.drop_elaboration_active() {
            return Vec::new();
        }
        let scope = self.drop_scopes.pop().expect("unbalanced drop scopes");
        if diverges {
            return Vec::new();
        }
        let names = scope.names.iter().rev().copied().collect();
        self.synthesize_drops(names, location)
    }

    /// Drop calls for a `return` edge: everything owned from the innermost scope up to and
    /// including the enclosing function-body scope (`return` exits the current lambda only).
    pub(super) fn drops_for_return(&mut self, location: &Location) -> Vec<ExprId> {
        if !self.drop_elaboration_active() {
            return Vec::new();
        }
        let mut names = Vec::new();
        for scope in self.drop_scopes.iter().rev() {
            names.extend(scope.names.iter().rev().copied());
            if scope.kind == DropScopeKind::Function {
                break;
            }
        }
        self.synthesize_drops(names, location)
    }

    /// Record `pattern`'s bindings as owned locals of the innermost drop scope.
    /// Bindings that alias a sub-place of another value (`binding_places`) are not owners
    /// and are skipped -- their root's owner drops them.
    pub(super) fn register_drop_locals(&mut self, pattern: PatternId) {
        if !self.drop_elaboration_active() || self.drop_scopes.is_empty() {
            return;
        }
        let mut names = Vec::new();
        self.collect_pattern_binding_names(pattern, &mut names);
        names.retain(|name| !self.binding_places.contains_key(name));
        let scope = self.drop_scopes.last_mut().unwrap();
        for name in names {
            if !scope.names.contains(&name) {
                scope.names.push(name);
            }
        }
    }

    fn collect_pattern_binding_names(&self, id: PatternId, out: &mut Vec<NameId>) {
        match self.pattern_of(id).as_ref() {
            cst::Pattern::Variable(name) | cst::Pattern::MethodName { item_name: name, .. } => {
                if !out.contains(name) {
                    out.push(*name);
                }
            },
            cst::Pattern::Alias(name, inner) => {
                if !out.contains(name) {
                    out.push(*name);
                }
                self.collect_pattern_binding_names(*inner, out);
            },
            cst::Pattern::TypeAnnotation(inner, _) => self.collect_pattern_binding_names(*inner, out),
            cst::Pattern::Or(alts) => {
                for alt in alts {
                    self.collect_pattern_binding_names(*alt, out);
                }
            },
            cst::Pattern::Constructor(_, args) => {
                for arg in args {
                    self.collect_pattern_binding_names(*arg, out);
                }
            },
            cst::Pattern::Literal(_) | cst::Pattern::Error => (),
        }
    }

    /// Filter `names` (already in drop order) down to those still owned and droppable here,
    /// and synthesize one drop call per survivor.
    fn synthesize_drops(&mut self, names: Vec<NameId>, location: &Location) -> Vec<ExprId> {
        let mut drops = Vec::new();
        for name in names {
            // Captured by some closure: the closure may outlive this scope; dropping the
            // referent would dangle it. Skip (leak) for now.
            if self.captured_names.contains(&name) {
                continue;
            }
            let place = MovePath::Variable(name);
            if self.move_tracker.is_moved(&place).is_some() {
                continue;
            }
            // Partially-moved: residual drops are a later increment. Skipping leaks the
            // remaining fields but never double-frees the moved ones.
            if self.move_tracker.has_child_moved(&place).is_some() {
                continue;
            }
            let Some(typ) = self.name_types.get(&name).cloned() else { continue };
            let typ = self.follow_type(&typ).clone();
            // Not fully concrete (generics included): dropping needs `{Drop t}` propagation.
            // We leak this for now.
            if typ == Type::ERROR || !typ.free_vars(&self.bindings).is_empty() {
                continue;
            }
            if self.type_is_copy(&typ) {
                continue;
            }
            // No visible Drop impl: skip (leak).
            if !self.type_has_drop_impl(&typ) {
                continue;
            }
            drops.push(self.synthesize_drop_call(name, &typ, location));
        }
        drops
    }

    /// Build and type-check `drop (mut <name>)` in the extended context, returning the call's
    /// `ExprId`. Checking it runs the full pipeline, so the `Drop` impl is resolved and
    /// materialized by implicit search (possibly delayed to the enclosing scope's pop) and the
    /// MIR builder can lower the expression like any user-written call.
    fn synthesize_drop_call(&mut self, name: NameId, typ: &Type, location: &Location) -> ExprId {
        let name_string = self.current_extended_context()[name].as_ref().clone();

        let place_path = self.push_path(
            cst::Path { components: vec![(name_string, location.clone())] },
            typ.clone(),
            location.clone(),
        );
        self.current_extended_context_mut().insert_path_origin(place_path, Origin::Local(name));
        let place_expr = self.push_expr(Expr::Variable(place_path), typ.clone(), location.clone());

        let ref_type = self.next_type_variable();
        let reference = cst::Reference { kind: ReferenceKind::Mut, rhs: place_expr };
        let ref_expr = self.push_expr(Expr::Reference(reference), ref_type, location.clone());

        let drop_name = self.get_drop_method_name();
        let callee_type = self.next_type_variable();
        let callee_path = self.push_path(
            cst::Path { components: vec![("drop".to_string(), location.clone())] },
            callee_type.clone(),
            location.clone(),
        );
        self.current_extended_context_mut().insert_path_origin(callee_path, Origin::TopLevelDefinition(drop_name));
        let callee = self.push_expr(Expr::Variable(callee_path), callee_type, location.clone());

        let call = cst::Call { function: callee, arguments: vec![cst::Argument::explicit(ref_expr)] };
        let call_expr = self.push_expr(Expr::Call(call), Type::UNIT, location.clone());

        // The value being dropped is dead on this edge; the synthesized use must neither
        // re-check nor re-record moves, and elaboration hooks must not observe this inference.
        let old_synthesizing = std::mem::replace(&mut self.synthesizing_drops, true);
        let old_check = std::mem::replace(&mut self.suppress_move_check, true);
        let old_record = std::mem::replace(&mut self.suppress_move_record, true);
        self.check_expr(call_expr, &Type::UNIT, TypeErrorKind::General);
        self.suppress_move_record = old_record;
        self.suppress_move_check = old_check;
        self.synthesizing_drops = old_synthesizing;

        call_expr
    }

    /// Returns the TopLevelName for the Prelude's `Drop.drop` method, caching it.
    pub(super) fn get_drop_method_name(&mut self) -> TopLevelName {
        if let Some(name) = self.drop_method_name {
            return name;
        }
        let exported = ExportedDefinitions(SourceFileId::prelude()).get(self.compiler);
        let name = exported.definitions.get(&Arc::new("drop".to_string())).expect("drop not found in Prelude");
        self.drop_method_name = Some(*name);
        *name
    }

    /// Over-approximate "some `Drop` impl is visible for `typ`" -- the same shape as
    /// [`TypeChecker::type_is_copy`], including its blindness to the candidate impl's own
    /// constraints. Used only to *skip* obligations that could not possibly resolve, so an
    /// over-approximation errs toward attempting resolution (whose failure is a real
    /// diagnostic), never toward a double-free.
    fn type_has_drop_impl(&mut self, typ: &Type) -> bool {
        let drop_type_name = self.get_drop_type_name();
        let constructor = Type::UserDefined(Origin::TopLevelDefinition(drop_type_name));
        let drop_of_t = Type::Application(Arc::new(constructor), Arc::new(vec![typ.clone()]));

        // Local implicits in scope (e.g. `{Drop t}` parameters).
        let local_implicits = self.collect_implicits_in_scope();
        for name in &local_implicits {
            let name_type = self.name_types[name].follow_all(&self.bindings);
            if self.try_unify(&name_type, &drop_of_t).is_ok() {
                return true;
            }
        }

        // Global impls visible from the current item's file.
        if let Some(item) = self.current_item {
            let visible_implicits = VisibleImplicits(item.source_file).get(self.compiler);
            let mut found = false;
            visible_implicits.iter_possibly_matching_impls(&drop_of_t, |_name, name_id| {
                let (name_type, _) = self.type_and_bindings_of_top_level_name(name_id);
                if self.try_unify(&name_type, &drop_of_t).is_ok() {
                    found = true;
                    return true;
                }
                if let Type::Function(f) = &name_type
                    && self.try_unify(&f.return_type, &drop_of_t).is_ok()
                {
                    found = true;
                    return true;
                }
                false
            });
            if found {
                return true;
            }
        }

        false
    }

    /// Returns the TopLevelName for the Prelude's `Drop` ability type, caching it.
    fn get_drop_type_name(&mut self) -> TopLevelName {
        if let Some(name) = self.drop_type_name {
            return name;
        }
        let exported_types = crate::incremental::ExportedTypes(SourceFileId::prelude()).get(self.compiler);
        let name = exported_types.get(&Arc::new("Drop".to_string())).expect("Drop type not found in Prelude");
        self.drop_type_name = Some(*name);
        *name
    }
}
