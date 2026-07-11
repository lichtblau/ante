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
    incremental::{ExportedDefinitions, GetItemRaw, VisibleImplicits},
    name_resolution::{Origin, namespace::SourceFileId},
    parser::{
        cst::{self, Expr, ReferenceKind},
        ids::{ExprId, NameId, PatternId, TopLevelId, TopLevelName},
    },
    type_inference::{
        TypeChecker,
        affine::{MovePath, MoveTracker},
        errors::TypeErrorKind,
        generics::Generic,
        types::{FunctionType, ParameterType, Type},
    },
};

/// The kind of scope a [`DropScope`] tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DropScopeKind {
    /// A block (`Expr::Sequence`); exits by fallthrough at its last statement.
    Block,
    /// A function body. Owns the parameters, and is the boundary `return` unwinds to.
    Function,
    /// A loop body (`while`/`for`); the boundary `break`/`continue` unwind to. Holds no
    /// droppable names of its own (the loop variable is an integer); body locals live in
    /// the body's own Block scope.
    LoopBody,
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
            if kind == DropScopeKind::Function {
                self.function_local_names.push(Default::default());
            }
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
        if scope.kind == DropScopeKind::Function {
            self.function_local_names.pop();
        }
        if diverges {
            return Vec::new();
        }
        let names = scope.names.iter().rev().copied().collect();
        self.synthesize_drops(names, location)
    }

    /// Drop calls for a `return` edge: everything owned from the innermost scope up to and
    /// including the enclosing function-body scope (`return` exits the current lambda only).
    pub(super) fn drops_for_return(&mut self, location: &Location) -> Vec<ExprId> {
        self.drops_up_to(DropScopeKind::Function, location)
    }

    /// Drop calls for a `break`/`continue` edge: everything owned from the innermost scope
    /// up to and including the innermost loop-body scope. `continue` needs the same set --
    /// it skips the rest of the body, including the body block's own fallthrough drops.
    pub(super) fn drops_for_loop_exit(&mut self, location: &Location) -> Vec<ExprId> {
        self.drops_up_to(DropScopeKind::LoopBody, location)
    }

    fn drops_up_to(&mut self, boundary: DropScopeKind, location: &Location) -> Vec<ExprId> {
        if !self.drop_elaboration_active() {
            return Vec::new();
        }
        let mut names = Vec::new();
        for scope in self.drop_scopes.iter().rev() {
            names.extend(scope.names.iter().rev().copied());
            if scope.kind == boundary {
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
        // `_` bindings are wildcards the MIR builder never materializes: a synthesized
        // `drop (mut _)` would reference an unbound variable. Their values leak in v1.
        names.retain(|name| self.current_extended_context()[*name].as_ref() != "_");
        if let Some(function_locals) = self.function_local_names.last_mut() {
            function_locals.extend(names.iter().copied());
        }
        let scope = self.drop_scopes.last_mut().unwrap();
        for name in names {
            if !scope.names.contains(&name) {
                scope.names.push(name);
            }
        }
    }

    pub(super) fn collect_pattern_binding_names(&self, id: PatternId, out: &mut Vec<NameId>) {
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
            let Some(typ) = self.name_types.get(&name).cloned() else { continue };
            let tracker = self.move_tracker.clone();
            if let Some(drop) = self.synthesize_partial_drop(&place, &typ, &tracker, location) {
                drops.push(drop);
            }
        }
        drops
    }

    /// Apply the per-place filters (captured root, concrete type, Copy, visible Drop impl)
    /// and synthesize the drop call if the place should be dropped. The caller has already
    /// decided the place is *owned* on the edge in question. `None` means skip (each skip
    /// can only leak, never double-free).
    pub(super) fn try_synthesize_drop_for_place(
        &mut self, place: &MovePath, typ: &Type, location: &Location,
    ) -> Option<ExprId> {
        // Captured by some closure: the closure may outlive this scope; dropping the
        // referent would dangle it. Skip (leak) for now.
        if self.captured_names.contains(&place.root_variable()) {
            return None;
        }
        // A borrowed alias binding elided its bind-retain; its scope-exit release is elided with it
        // (the pair must stay balanced).
        if self.borrowed_bindings.contains_key(&place.root_variable()) {
            return None;
        }
        // A borrowing parameter is caller-owned -- the call site either elided the retain or
        // balances it with a post-call release; never release here.
        if self.borrowed_local_params.contains(&place.root_variable()) {
            return None;
        }
        // A `move` closure's env owns its non-Copy captures. When the closure value itself dies
        // un-moved (this edge -- `synthesize_partial_drop` already returned early if it escaped),
        // drop each captured slot. Runs before the `Copy` fast path below: a closure value is
        // Copy-by-fiat, so it would otherwise synthesize nothing.
        if let MovePath::Variable(binding) = place
            && self.shared_closure_captures.contains_key(binding)
        {
            return self.synthesize_closure_binding_env_drops(*binding, place, typ, location);
        }
        let typ = self.follow_type(typ).clone();
        if typ == Type::ERROR {
            return None;
        }
        // Honest for bare variables under the flag: literal vars and `{Copy t}`-bounded
        // generics are Copy; everything else falls through to the generic rules below.
        if self.type_is_copy(&typ) {
            // Shared handles are Copy but reference-counted: a binding's death releases its
            // reference via the type's synthesized `release_T`.
            if self.is_shared_user_defined(&typ) {
                return Some(self.synthesize_release_call(place, &typ, location));
            }
            // Heap-env closures are Copy but their environment is reference-counted:
            // A binding's death releases the env's refcount, freeing the block at zero.
            // Effect continuations (`resume`) share the shape but have coroutine-supplied, non-RC
            // environments -- do not release them yet.
            if self.is_heap_env_closure(&typ) && !self.effect_continuation_names.contains(&place.root_variable()) {
                return Some(self.synthesize_closure_env_release(place, &typ, location));
            }
            // A Copy aggregate can still transitively own shared fields (a struct of all-Copy
            // fields is itself Copy). Structural drop reaches those fields (each shared one hits
            // the branch above) and self-prunes to `None` when nothing inside needs releasing --
            // so pure-Copy values (primitives, `{Copy t}` generics) still produce no drop.
            return self.synthesize_structural_drop(place, &typ, location);
        }
        // A non-`Copy` closure is *affine* -- its env tuple owns at least one non-`Copy` capture,
        // and the owner travels with the closure value. Tear the env down by extracting each owned
        // slot out of the closure and dropping it.
        if let Type::Function(_) = &typ {
            let drops = self.synthesize_owned_env_slot_drops(place, &typ, location);
            return self.sequence_drops(drops, location);
        }
        // A rigid generic: a named signature generic (`Type::Generic`, from annotated
        // signatures) or a bare type variable connected to the signature (unannotated
        // ones). Literal vars and `{Copy t}`-bounded generics returned Copy above;
        // in-flight unification variables are skipped silently. Drop through an in-scope
        // `{Drop t}` capability, or demand one -- the strict rule.
        if let Type::Generic(_) = &typ {
            let drop_ability = self.get_drop_type_name();
            if self.constraint_in_scope_for_generic(&typ, drop_ability) {
                return Some(self.synthesize_drop_call(place, &typ, location));
            }
            self.report_missing_drop_constraint(place, &typ, location);
            return None;
        }
        if let Type::Variable(id) = &typ {
            let drop_ability = self.get_drop_type_name();
            if self.constraint_in_scope_for_variable(*id, drop_ability) {
                return Some(self.synthesize_drop_call(place, &typ, location));
            }
            if self.is_signature_variable(*id) {
                self.report_missing_drop_constraint(place, &typ, location);
            }
            return None;
        }
        // Partially generic (`Vec u`, `Maybe u`): resolve through an impl when one exists
        // and every free variable has its `{Drop v}` in scope (guaranteed resolvable);
        // otherwise derive structurally -- recursion reaches the bare-variable rules above
        // for generic components (`Maybe u`'s payload), and prunes what needs nothing.
        // In-flight element variables that never get constrained are left alone (leak)
        // rather than erroring on dead values like a never-pushed `Vec.empty ()`.
        if !typ.free_vars(&self.bindings).is_empty() {
            let drop_ability = self.get_drop_type_name();
            let free_vars = typ.free_vars(&self.bindings);
            let all_vars_bounded = free_vars.iter().all(|generic| {
                let target = match generic {
                    super::generics::Generic::Inferred(id) => {
                        // A numeric-literal var (`var v = Vec.empty (); v.push 1`) defaults
                        // to I32/F64 by item end: synthesize the drop and let its implicit
                        // Drop search resolve against the defaulted element type.
                        if self.integer_literal_vars.contains(id) || self.float_literal_vars.contains(id) {
                            return true;
                        }
                        Type::Variable(*id)
                    },
                    named => Type::Generic(named.clone()),
                };
                self.constraint_in_scope_for_generic(&target, drop_ability)
            });
            if all_vars_bounded && self.type_has_drop_impl(&typ) {
                return Some(self.synthesize_drop_call(place, &typ, location));
            }
            return self.synthesize_structural_drop(place, &typ, location);
        }
        // No direct impl: derive a structural drop -- for concrete product types, drop
        // each field that needs one, in declaration order. `None` when no component needs a
        // drop, so `needs_drop` pruning falls out naturally.
        if !self.type_has_drop_impl(&typ) {
            return self.synthesize_structural_drop(place, &typ, location);
        }
        Some(self.synthesize_drop_call(place, &typ, location))
    }

    /// The strict `{Drop t}` rule: a value of bare generic type dies here with no way to
    /// drop it. Reported once per place, pointing at the value's binding.
    fn report_missing_drop_constraint(&mut self, place: &MovePath, typ: &Type, fallback: &Location) {
        if self.suppress_missing_drop_diagnostic {
            return;
        }
        if !self.diagnosed_missing_drops.insert(place.clone()) {
            return;
        }
        let root = place.root_variable();
        let location = {
            use crate::parser::ids::NameStore;
            if self.current_extended_context().try_get_name(root).is_some() {
                self.current_extended_context().name_location(root)
            } else {
                fallback.clone()
            }
        };
        let typ = self.type_to_string(typ);
        self.compiler.accumulate(crate::diagnostics::Diagnostic::MissingDropConstraint { typ, location });
    }

    /// Derive the structural drop for a product type as an inline `Sequence` of per-field
    /// drops (declaration order), recursing through [`Self::try_synthesize_drop_for_place`]
    /// so nested fields with real impls call them and no-op components prune away. A depth
    /// cap guards recursive types (e.g. through `Maybe`), which cannot expand inline --
    /// capped components are skipped (leak, never double-free).
    fn synthesize_structural_drop(&mut self, place: &MovePath, typ: &Type, location: &Location) -> Option<ExprId> {
        if self.drop_expansion_depth >= 16 {
            return None;
        }
        self.drop_expansion_depth += 1;
        let result = self.synthesize_structural_drop_inner(place, typ, location);
        self.drop_expansion_depth -= 1;
        result
    }

    fn synthesize_structural_drop_inner(
        &mut self, place: &MovePath, typ: &Type, location: &Location,
    ) -> Option<ExprId> {
        let fields = self.get_field_types(typ, None);
        if fields.is_empty() {
            // Not a product: derive a sum drop (an exhaustive match dropping each variant's
            // payloads) if it is an enum; anything else is skipped (leak).
            return self.synthesize_sum_drop(place, typ, None, location);
        }
        let mut ordered: Vec<(String, Type, u32)> =
            fields.into_iter().map(|(name, (typ, index))| (name.to_string(), typ, index)).collect();
        ordered.sort_unstable_by_key(|(_, _, index)| *index);

        let mut drops = Vec::new();
        for (field_name, field_type, _) in ordered {
            let field_place = MovePath::field(place.clone(), field_name);
            if let Some(drop) = self.try_synthesize_drop_for_place(&field_place, &field_type, location) {
                drops.push(drop);
            }
        }
        if drops.is_empty() {
            return None;
        }
        let items = drops.into_iter().map(|expr| cst::SequenceItem { comments: Vec::new(), expr }).collect();
        Some(self.push_expr(Expr::Sequence(items), Type::UNIT, location.clone()))
    }

    /// For each `if`/`match` merge edge, drop the whole locals that a *sibling* branch moved
    /// but this edge still owns. The merged tracker reads such places as moved (union), so
    /// the end of the still-owning branch is the only point that can release them.
    /// `edges` holds each non-diverging branch's body expression and end-state tracker.
    /// Field paths (partial moves) are deferred to the residual increment.
    pub(super) fn equalize_branch_drops(&mut self, pre_branch: &MoveTracker, edges: &[(ExprId, MoveTracker)]) {
        if !self.drop_elaboration_active() || edges.is_empty() {
            return;
        }
        let candidates = self.branch_move_candidates(pre_branch, edges.iter().map(|(_, tracker)| tracker));
        if candidates.is_empty() {
            return;
        }
        for (edge_expr, tracker) in edges {
            let location = self.current_extended_context().expr_location(*edge_expr);
            let mut drops = Vec::new();
            for name in &candidates {
                let place = MovePath::Variable(*name);
                let Some(typ) = self.name_types.get(name).cloned() else { continue };
                if let Some(drop) = self.synthesize_partial_drop(&place, &typ, tracker, &location) {
                    drops.push(drop);
                }
            }
            if !drops.is_empty() {
                self.current_extended_context_mut().push_post_expr_drops(*edge_expr, drops);
            }
        }
    }

    /// Drops for the implicit else edge of an else-less `if` whose then-branch moved values:
    /// on the false edge those values are still owned, and there is no expression to key them
    /// on -- the builder materializes a real else block from `implicit_else_drops`.
    pub(super) fn implicit_else_edge_drops(
        &mut self, pre_branch: &MoveTracker, then_moves: &MoveTracker, location: &Location,
    ) -> Vec<ExprId> {
        if !self.drop_elaboration_active() {
            return Vec::new();
        }
        let candidates = self.branch_move_candidates(pre_branch, std::iter::once(then_moves));
        let mut drops = Vec::new();
        for name in candidates {
            // The implicit edge's state is `pre_branch`, which owns every candidate.
            let Some(typ) = self.name_types.get(&name).cloned() else { continue };
            let place = MovePath::Variable(name);
            if let Some(drop) = self.synthesize_partial_drop(&place, &typ, pre_branch, location) {
                drops.push(drop);
            }
        }
        drops
    }

    /// Whole locals declared in an enclosing drop scope that some branch moved while the
    /// pre-branch state still owned them -- the candidate set for edge equalization. Ordered
    /// innermost-scope-first, reverse declaration order (the drop order used on each edge).
    fn branch_move_candidates<'a>(
        &self, pre_branch: &MoveTracker, trackers: impl Iterator<Item = &'a MoveTracker>,
    ) -> Vec<NameId> {
        let mut moved_somewhere = rustc_hash::FxHashSet::default();
        for tracker in trackers {
            for name in tracker.whole_moved_locals() {
                if pre_branch.is_moved(&MovePath::Variable(name)).is_none() {
                    moved_somewhere.insert(name);
                }
            }
        }
        if moved_somewhere.is_empty() {
            return Vec::new();
        }
        let mut ordered = Vec::new();
        for scope in self.drop_scopes.iter().rev() {
            for name in scope.names.iter().rev() {
                // Only this function's own locals: equalizing a captured outer variable
                // would drop through the closure's capture reference.
                if moved_somewhere.contains(name) && self.name_is_local_to_current_function(*name) {
                    ordered.push(*name);
                }
            }
        }
        ordered
    }

    /// `x := new` over a live owned place drops the old value after the
    /// RHS is evaluated, before the store. Restricted to places rooted
    /// at owned locals -- deref-stores through references or pointers
    /// (`ptr_store`) never drop the old pointee (documented v1 parity
    /// gap). Must run before `clear_moves` re-marks the place owned.
    pub(super) fn assignment_overwrite_drop(&mut self, assignment: &cst::Assignment, id: ExprId) {
        if !self.drop_elaboration_active() {
            return;
        }
        let Some(place) = self.try_build_move_path(assignment.lhs) else { return };
        let root = place.root_variable();
        // Assigning to a captured outer variable writes through the closure's capture reference:
        // dropping the old value there is a through-reference drop. Excluded in v1 -- EXCEPT inside
        // a handler-scoped lambda, which provably cannot outlive its handle expression: the
        // captured slot then outlives every run of this lambda, so releasing the overwritten value
        // through the capture is sound. This is the foldl-accumulator fix (`shared.an`): `result :=
        // f result value` in the emit handler now releases each old accumulator instead of
        // orphaning it. Ordinary (escapable) closures keep the v1 skip.
        let through_handler_capture =
            self.in_handler_scoped_lambda && self.name_is_local_to_any_enclosing_function(root);
        if !self.name_is_local_to_current_function(root) && !through_handler_capture {
            return;
        }
        let Some(root_type) = self.name_types.get(&root).cloned() else { return };
        if root_type.reference_element(&self.bindings).is_some()
            || root_type.pointer_element(&self.bindings).is_some()
        {
            return;
        }
        let Some(lhs_type) = self.expr_types.get(&assignment.lhs).cloned() else { return };
        let location = self.current_extended_context().expr_location(assignment.lhs);
        let tracker = self.move_tracker.clone();
        // Overwritten generic places often hold bit-copies of values another structure
        // owns; skip silently rather than demand a bound that would double-free.
        let old_suppress = std::mem::replace(&mut self.suppress_missing_drop_diagnostic, true);
        let drop = self.synthesize_partial_drop(&place, &lhs_type, &tracker, &location);
        self.suppress_missing_drop_diagnostic = old_suppress;
        if let Some(drop) = drop {
            self.current_extended_context_mut().push_pre_exit_drops(id, vec![drop]);
        }
    }

    /// If the just-inferred, non-final statement `item` produced a discarded owned value,
    /// rewrite it in place to `tmp = <item>; drop (mut tmp)` so the temporary drops at
    /// statement end (Rust parity). The statement was already inferred -- its content is
    /// copied to a fresh id (metadata included) rather than re-inferred, so no moves are
    /// double-recorded.
    pub(super) fn drop_discarded_statement_value(&mut self, item: ExprId, typ: &Type) {
        if !self.drop_elaboration_active() || self.diverges(typ) {
            return;
        }
        let typ = self.follow_type(typ).clone();
        if typ == Type::ERROR || !typ.free_vars(&self.bindings).is_empty() {
            return;
        }
        if self.type_is_copy(&typ) {
            return;
        }

        let location = self.current_extended_context().expr_location(item);

        // The temporary's drop goes through the unified synthesis (real impl or derived
        // structural/sum drop). If nothing needs dropping, leave the statement alone.
        let (_tmp_path, tmp_name) = self.fresh_variable("drop_tmp", typ.clone(), location.clone());
        self.name_types.insert(tmp_name, typ.clone());
        let Some(drop_call) = self.try_synthesize_drop_for_place(&MovePath::Variable(tmp_name), &typ, &location)
        else {
            return;
        };

        // Copy the statement's content to a fresh id (we are about to replace its own id),
        // keeping per-expr metadata (decision trees, member indices, drop tables) -- then
        // clear the drop tables at the outer id so they cannot fire twice.
        let content = match self.current_extended_context().extended_expr(item) {
            Some(expr) => expr.clone(),
            None => self.current_context()[item].clone(),
        };
        let copied = self.push_expr(content, typ.clone(), location.clone());
        self.current_extended_context_mut().copy_expr_metadata(item, copied);
        self.current_extended_context_mut().clear_expr_drops(item);

        // `tmp = <copied>; drop (mut tmp)`
        let definition = self.let_binding(tmp_name, copied);

        let seq_item = |expr| cst::SequenceItem { comments: Vec::new(), expr };
        let block = Expr::Sequence(vec![seq_item(definition), seq_item(drop_call)]);
        self.current_extended_context_mut().insert_expr(item, block);
        self.expr_types.insert(item, Type::UNIT);
    }

    /// Decide which explicit parameters of a top-level function are *borrowing* -- the callee never
    /// releases them; statically-known call sites either elide the argument retain (pure implicits:
    /// the callee provably cannot suspend) or balance it with a post-call release (the caller's
    /// frame then holds the handle across any suspension). Runs BEFORE the body is inferred -- the
    /// analysis is purely syntactic over resolved names -- so the release skip is in place when the
    /// function scope pops.
    ///
    /// A parameter is borrowing iff it is by-value (immutable), its annotated type is
    /// concretely `shared`, and the body never captures it in a nested lambda (tail/return
    /// references are fine -- see [`Self::param_escapes_in_body`]). The whole function is
    /// disqualified when any explicit parameter is function- or ability-typed: calling
    /// those is a suspension avenue the call site cannot vet (implicit capabilities are
    /// vetted per call site instead).
    pub(super) fn compute_borrowed_param_mask(&mut self, fn_name: NameId, lambda: &cst::Lambda, expected: &Type) {
        if !self.drop_elaboration_active() {
            return;
        }
        let expected = self.follow_type(expected).clone();
        let Type::Function(function_type) = &expected else { return };
        if function_type.parameters.len() != lambda.parameters.len() {
            return;
        }
        for (param, param_type) in lambda.parameters.iter().zip(&function_type.parameters) {
            if param.is_implicit {
                continue;
            }
            let typ = self.follow_type(&param_type.typ).clone();
            if matches!(typ, Type::Function(_)) || self.is_ability(&typ) {
                return;
            }
        }

        let mut mask = Vec::new();
        for (param, param_type) in lambda.parameters.iter().zip(&function_type.parameters) {
            if param.is_implicit {
                continue;
            }
            let typ = self.follow_type(&param_type.typ).clone();
            let mut borrowing = false;
            if !param.is_mutable && self.is_shared_user_defined(&typ) {
                if let Some(name) = self.single_variable_pattern(param.pattern) {
                    if !self.param_escapes_in_body(lambda.body, name) {
                        borrowing = true;
                        self.borrowed_local_params.insert(name);
                    }
                }
            }
            mask.push(borrowing);
        }
        if mask.iter().any(|b| *b)
            && let Some(item) = self.current_item
        {
            self.borrowed_param_masks.entry(item).or_default().insert(fn_name, mask);
        }
    }

    /// The single variable a pattern binds (through type annotations), if it is that simple.
    pub(super) fn single_variable_pattern(&self, pattern: PatternId) -> Option<NameId> {
        match self.pattern_of(pattern).as_ref() {
            cst::Pattern::Variable(name) => Some(*name),
            cst::Pattern::TypeAnnotation(inner, _) => self.single_variable_pattern(*inner),
            _ => None,
        }
    }

    /// The name a top-level definition binds, including `MethodName` patterns (`Vec.get`,
    /// `HashMap.get`) -- `single_variable_pattern` deliberately stops at those, but the
    /// return-origin summary wants to key methods too. Used only to name a summary side-table row.
    pub(super) fn definition_binding_name(&self, pattern: PatternId) -> Option<NameId> {
        match self.pattern_of(pattern).as_ref() {
            cst::Pattern::Variable(name) | cst::Pattern::MethodName { item_name: name, .. } => Some(*name),
            cst::Pattern::TypeAnnotation(inner, _) => self.definition_binding_name(*inner),
            _ => None,
        }
    }

    /// True when `name` (a parameter) escapes the function through `expr` by being CAPTURED
    /// by a nested lambda (the closure's bit-copy has no count of its own and may outlive
    /// the call). Tail/return references are NOT escapes: a returned shared place gets an
    /// escape retain (`mark_tail_escape_retains`, applied to lambda-body tails and explicit
    /// `return`s alike), so the returned copy carries its own count -- the exact invariant
    /// callee-owns parameters already rely on when they are returned. Call arguments, field
    /// reads, and stores each carry their own balanced retain.
    fn param_escapes_in_body(&self, expr: ExprId, name: NameId) -> bool {
        let walk = |this: &Self, e: ExprId| this.param_escapes_in_body(e, name);
        match self.expr_of(expr).as_ref() {
            Expr::Lambda(_) => self.lambda_captures_name(expr, name),
            Expr::Error
            | Expr::Variable(_)
            | Expr::Literal(_)
            | Expr::Extern(_)
            | Expr::Break
            | Expr::Continue
            | Expr::Quoted(_) => false,
            Expr::Sequence(items) => items.iter().any(|item| walk(self, item.expr)),
            Expr::Definition(definition) => walk(self, definition.rhs),
            Expr::MemberAccess(access) => walk(self, access.object),
            Expr::Call(call) => {
                walk(self, call.function) || call.arguments.iter().any(|arg| walk(self, arg.expr))
            },
            Expr::If(if_) => {
                walk(self, if_.condition) || walk(self, if_.then) || if_.else_.is_some_and(|e| walk(self, e))
            },
            Expr::Match(match_) => {
                walk(self, match_.expression) || match_.cases.iter().any(|(_, branch)| walk(self, *branch))
            },
            // A handle's body and branches are lambdas; the Lambda arm above applies to them.
            Expr::Handle(handle) => {
                walk(self, handle.expression) || handle.cases.iter().any(|(_, branch)| walk(self, *branch))
            },
            Expr::Reference(reference) => walk(self, reference.rhs),
            Expr::TypeAnnotation(annotation) => walk(self, annotation.lhs),
            Expr::Constructor(constructor) => constructor.fields.iter().any(|(_, e)| walk(self, *e)),
            Expr::While(w) => walk(self, w.condition) || walk(self, w.body),
            Expr::For(f) => walk(self, f.start) || walk(self, f.end) || walk(self, f.body),
            Expr::Return(return_) => walk(self, return_.expression),
            Expr::Assignment(assignment) => {
                walk(self, assignment.lhs)
                    || walk(self, assignment.rhs)
                    || assignment.op.as_ref().is_some_and(|(_, op)| walk(self, *op))
            },
            Expr::ArrayLiteral(elements) => elements.iter().any(|e| walk(self, *e)),
            // Not yet desugared here (or unexpected): be conservative.
            Expr::Is(_) | Expr::Do(_) | Expr::Loop(_) | Expr::InterpolatedString(_) => true,
        }
    }

    /// `definition` is a borrowed alias binding -- a single un-annotated, immutable variable bound
    /// to a *whole-place read of another immutable local* -- when its retain+release pair can be
    /// elided. The alias's scope is lexically inside the source's and an immutable source is only
    /// released at its own scope exit, so the handle outlives every use. `var` sources or field
    /// reads keep the pair: `:=` through them releases the old value while the alias still points
    /// at it.
    pub(super) fn try_borrowed_alias_binding(&mut self, definition: &cst::Definition) -> Option<NameId> {
        if definition.mutable || !self.drop_elaboration_active() {
            return None;
        }
        let bound = match self.pattern_of(definition.pattern).as_ref() {
            cst::Pattern::Variable(name) => *name,
            _ => return None,
        };
        let rhs = self.current_extended_context()[definition.rhs].clone();
        let Expr::Variable(path) = rhs else { return None };
        let Some(Origin::Local(source)) = self.path_origin(path) else { return None };
        if self.mutable_definitions.contains(&source) {
            return None;
        }
        Some(bound)
    }

    /// Auto-ref of an *rvalue* in call-argument position (`println ("a" ++ "b")`): the callee
    /// only borrows through the reference, so the caller owns the temporary -- but no scope
    /// ever dropped it (the string-interpolation leak class). Bind the rvalue to a fresh
    /// local inside the reference (`ref (tmp = <rvalue>; tmp)`) and queue `drop (mut tmp)`
    /// for [`infer_call`] to attach as a post-expr drop on the enclosing call, so the
    /// temporary dies right after the call returns. Places are untouched (their owner drops
    /// at scope exit -- the auto-ref move-restore already handles them); Copy or otherwise
    /// undroppable types synthesize no drop and stay unwrapped.
    pub(super) fn bind_autoref_temp_for_drop(&mut self, rhs: ExprId, element_type: &Type, location: &Location) -> ExprId {
        if self.call_argument_depth == 0 || !self.drop_elaboration_active() {
            return rhs;
        }
        // Only rvalues: a place already has an owner.
        if self.try_build_move_path(rhs).is_some() {
            return rhs;
        }
        let typ = self.follow_type(element_type).clone();
        if typ == Type::ERROR || !typ.free_vars(&self.bindings).is_empty() {
            return rhs;
        }
        let (_tmp_path, tmp_name) = self.fresh_variable("autoref_tmp", typ.clone(), location.clone());
        self.name_types.insert(tmp_name, typ.clone());
        let Some(drop_call) = self.try_synthesize_drop_for_place(&MovePath::Variable(tmp_name), &typ, location)
        else {
            return rhs;
        };
        let tmp_var = self.synthesize_place_expr(&MovePath::Variable(tmp_name), &typ, location);
        let block = self.let_binding_and_body(tmp_name, rhs, tmp_var);
        self.pending_autoref_temp_drops.push(drop_call);
        block
    }

    /// Build and type-check `drop (mut <place>)` in the extended context, returning the call's
    /// `ExprId`. Checking it runs the full pipeline, so the `Drop` impl is resolved and
    /// materialized by implicit search (possibly delayed to the enclosing scope's pop) and the
    /// MIR builder can lower the expression like any user-written call.
    fn synthesize_drop_call(&mut self, place: &MovePath, typ: &Type, location: &Location) -> ExprId {
        let place_expr = self.synthesize_place_expr(place, typ, location);

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

    /// Build the expression denoting `place`: a variable reference for a root, member-access
    /// chains for field paths (`x.field`). Only real struct fields appear here -- enum payload
    /// places (`Ok#0`) have no expression form and never reach this (residual drops of sums
    /// synthesize a match instead).
    fn synthesize_place_expr(&mut self, place: &MovePath, typ: &Type, location: &Location) -> ExprId {
        match place {
            MovePath::Variable(name) => {
                let name_string = self.current_extended_context()[*name].as_ref().clone();
                let place_path = self.push_path(
                    cst::Path { components: vec![(name_string, location.clone())] },
                    typ.clone(),
                    location.clone(),
                );
                self.current_extended_context_mut().insert_path_origin(place_path, Origin::Local(*name));
                self.push_expr(Expr::Variable(place_path), typ.clone(), location.clone())
            },
            MovePath::Field(parent, field) => {
                let parent_type = match parent.as_ref() {
                    MovePath::Variable(name) => {
                        self.name_types.get(name).cloned().unwrap_or_else(|| self.next_type_variable())
                    },
                    MovePath::Field(..) => self.next_type_variable(),
                };
                let object = self.synthesize_place_expr(parent, &parent_type, location);
                let access = cst::MemberAccess { object, member: field.clone() };
                self.push_expr(Expr::MemberAccess(access), typ.clone(), location.clone())
            },
        }
    }

    /// The stable [TopLevelName] of a shared type's synthesized `release_T` function.  Both a
    /// release call site and the type's own MIR emission derive it from the type's item id, so
    /// `name_to_id`/`get_definition_id` resolve them to one definition.
    pub(super) fn get_release_function_name(&self, type_id: TopLevelId) -> TopLevelName {
        TopLevelName::new(type_id, NameId::RELEASE_FUNCTION)
    }

    pub(super) fn is_release_function_name(name: TopLevelName) -> bool {
        name.local_name_id == NameId::RELEASE_FUNCTION
    }

    /// The generalized type of a shared type's `release_T`: `forall <generics>. <SharedType> -> Unit`
    /// (the handle is passed by value -- it is a Copy pointer). `release_T` is not a parsed item, so
    /// inference computes its type here on demand ([Self::type_of_top_level_name] intercepts the
    /// reserved name), which lets release calls type-check and instantiate like any generic call.
    pub(super) fn release_function_generalized_type(&self, type_id: TopLevelId) -> Type {
        let (item, _) = GetItemRaw(type_id).get(self.compiler);
        let cst::TopLevelItemKind::TypeDefinition(td) = &item.kind else {
            return Type::ERROR;
        };
        let type_name = TopLevelName::new(type_id, td.name);
        let mut data_type = Type::UserDefined(Origin::TopLevelDefinition(type_name));
        let generics: Vec<Generic> = td.generics.iter().map(|p| Generic::Named(Origin::Local(p.name))).collect();
        if !generics.is_empty() {
            let generic_types = generics.iter().map(|g| Type::Generic(g.clone())).collect::<Vec<_>>();
            data_type = Type::Application(Arc::new(data_type), Arc::new(generic_types));
        }
        let fn_type = Type::Function(Arc::new(FunctionType {
            parameters: vec![ParameterType::explicit(data_type)],
            environment: Type::NO_CLOSURE_ENV,
            return_type: Type::UNIT,
        }));
        if generics.is_empty() { fn_type } else { Type::Forall(Arc::new(generics), Arc::new(fn_type)) }
    }

    /// Synthesize a call `release_T <place>` releasing a shared handle by value.
    /// Unlike [Self::synthesize_drop_call] it passes the handle directly (a Copy pointer, no `mut`
    /// reference). The callee resolves through the reserved release name; `check_expr` types and
    /// instantiates it exactly like a real generic call.
    fn synthesize_release_call(&mut self, place: &MovePath, typ: &Type, location: &Location) -> ExprId {
        let place_expr = self.synthesize_place_expr(place, typ, location);

        let type_id = self
            .shared_type_top_level_id(typ)
            .expect("synthesize_release_call: place is not a shared user-defined type");
        let release_name = self.get_release_function_name(type_id);
        let callee_type = self.next_type_variable();
        let callee_path = self.push_path(
            cst::Path { components: vec![("release".to_string(), location.clone())] },
            callee_type.clone(),
            location.clone(),
        );
        self.current_extended_context_mut().insert_path_origin(callee_path, Origin::TopLevelDefinition(release_name));
        let callee = self.push_expr(Expr::Variable(callee_path), callee_type, location.clone());

        let call = cst::Call { function: callee, arguments: vec![cst::Argument::explicit(place_expr)] };
        let call_expr = self.push_expr(Expr::Call(call), Type::UNIT, location.clone());

        let old_synthesizing = std::mem::replace(&mut self.synthesizing_drops, true);
        let old_check = std::mem::replace(&mut self.suppress_move_check, true);
        let old_record = std::mem::replace(&mut self.suppress_move_record, true);
        self.check_expr(call_expr, &Type::UNIT, TypeErrorKind::General);
        self.suppress_move_record = old_record;
        self.suppress_move_check = old_check;
        self.synthesizing_drops = old_synthesizing;

        call_expr
    }

    /// Synthesize a heap-env closure's environment release. Unlike a shared handle (a nominal
    /// `release_T` call) closures have no nominal release function -- the operation is uniform
    /// (decrement the env pointer's refcount, free at zero), so the place expression is recorded
    /// directly in the `closure_env_releases` side table and the MIR builder emits a null-safe
    /// `ReleaseClosureEnv` on the place's env pointer. The place expression is checked with moves
    /// suppressed (the closure value is dead on this edge; it must neither re-check nor record).
    fn synthesize_closure_env_release(&mut self, place: &MovePath, typ: &Type, location: &Location) -> ExprId {
        let place_expr = self.synthesize_place_expr(place, typ, location);

        let old_synthesizing = std::mem::replace(&mut self.synthesizing_drops, true);
        let old_check = std::mem::replace(&mut self.suppress_move_check, true);
        let old_record = std::mem::replace(&mut self.suppress_move_record, true);
        self.check_expr(place_expr, typ, TypeErrorKind::General);
        self.suppress_move_record = old_record;
        self.suppress_move_check = old_check;
        self.synthesizing_drops = old_synthesizing;

        self.current_extended_context_mut().mark_closure_env_release(place_expr);
        place_expr
    }

    /// Synthesize the env drop for a dying `move` closure `binding` -- one drop per non-Copy,
    /// non-`var` capture, in capture order. Each capture aliases its env slot (a `move` closure
    /// holds the captured value by value, `pack_closure_environment`), so dropping it by its own
    /// name frees the env's heap; `captured_names` normally suppresses that drop, so it is
    /// force-enabled for the capture's whole subtree while its drop is synthesized. Returns a
    /// `Sequence` of the drops, or `None` when every capture needs nothing (Copy prunes away).
    fn synthesize_closure_binding_env_drops(
        &mut self, binding: NameId, place: &MovePath, typ: &Type, location: &Location,
    ) -> Option<ExprId> {
        let typ = self.follow_type(typ).clone();
        let mut drops = self.synthesize_owned_env_slot_drops(place, &typ, location);
        drops.extend(self.synthesize_shared_env_release(binding, location));
        self.sequence_drops(drops, location)
    }

    /// Wrap a teardown's drop calls in a `Sequence`, or `None` when there is nothing to do.
    fn sequence_drops(&mut self, drops: Vec<ExprId>, location: &Location) -> Option<ExprId> {
        if drops.is_empty() {
            return None;
        }
        let items = drops.into_iter().map(|expr| cst::SequenceItem { comments: Vec::new(), expr }).collect();
        Some(self.push_expr(Expr::Sequence(items), Type::UNIT, location.clone()))
    }

    /// The closure destructor. One `$slot = <env slot i of place>; drop (mut $slot)`
    /// pair per owned (non-`Copy`, non-ref) slot of the dying closure's env tuple, in slot order.
    ///
    /// The env slot is extracted from the **closure value** rather than dropped through the
    /// capture's original name, because the closure may be dying in a function that never saw those
    /// names -- the escaping case this task exists to fix. The frontend cannot *write* the extract
    /// (a closure's env has no surface syntax), so each place expression is tagged in the
    /// `closure_env_slots` side table and the MIR builder rewrites it into `IndexTuple` pairs.
    ///
    /// Slot selection mirrors [Self::env_type_is_copy], the same predicate that made the closure
    /// affine in the first place: refs (`var` Mut-refs, Part-B `IMM` borrows), `Copy` captures
    /// (primitives, `shared` handles, free functions) and unknown/generic envs are skipped, so a
    /// closure reaching here has at least one owned slot. Skipping can only leak, never double-free.
    fn synthesize_owned_env_slot_drops(
        &mut self, place: &MovePath, typ: &Type, location: &Location,
    ) -> Vec<ExprId> {
        let Type::Function(function) = self.follow_type(typ).clone() else { return Vec::new() };
        // A heap env is an opaque `Pointer` at the type layer -- its slots are unknowable here, and
        // its refcount release is the Copy path's `ReleaseClosureEnv`.
        let Type::Tuple(slots) = self.follow_type(&function.environment).clone() else { return Vec::new() };

        let mut drops = Vec::new();
        for (index, slot_type) in slots.iter().enumerate() {
            let slot_type = self.follow_type(slot_type).clone();
            if self.env_type_is_copy(&slot_type) {
                continue;
            }
            let (_slot_path, slot_name) = self.fresh_variable("env_slot", slot_type.clone(), location.clone());
            self.name_types.insert(slot_name, slot_type.clone());
            let Some(drop_call) =
                self.try_synthesize_drop_for_place(&MovePath::Variable(slot_name), &slot_type, location)
            else {
                continue;
            };

            // `$env_slot = <place>` -- the rhs is a plain place expression until the MIR builder
            // reads the marker and lowers it as an env-slot extract of the closure's value.
            let extract = self.synthesize_place_expr(place, typ, location);
            self.expr_types.insert(extract, slot_type);
            self.current_extended_context_mut().mark_closure_env_slot(extract, index as u32);

            drops.push(self.let_binding(slot_name, extract));
            drops.push(drop_call);
        }
        drops
    }

    /// Release each `shared` capture of a dying closure `binding` -- the env's own reference,
    /// balancing the pack-time `RcRetain`. Unlike the owned move-env drop these captures were
    /// already removed from `captured_names` (`record_shared_captures`, the owner restore), so the
    /// blanket does not suppress them; `try_synthesize_drop_for_place` resolves each shared type to
    /// its `release_T`. Fires for `move` and non-`move` closures alike. These are dropped **by
    /// name**, so they only fire in the defining scope -- a shared-only closure is Copy, so it
    /// escapes by bit-copy and is retracted instead.
    fn synthesize_shared_env_release(&mut self, binding: NameId, location: &Location) -> Vec<ExprId> {
        let Some(captures) = self.shared_closure_captures.get(&binding).cloned() else { return Vec::new() };
        let mut drops = Vec::new();
        for capture in captures {
            let Some(typ) = self.name_types.get(&capture).cloned() else { continue };
            if let Some(drop) = self.try_synthesize_drop_for_place(&MovePath::Variable(capture), &typ, location) {
                drops.push(drop);
            }
        }
        drops
    }

    /// A `move` closure is Copy-by-fiat (`affine.rs`), so returning, storing, or passing it records
    /// **no** move -- its by-value env tuple is bit-copied to the escaping location and the copy
    /// aliases the captured values. Dropping those captures when the closure's own scope exits
    /// would then free heap the escaped copy still points at (a use-after-free). Before a scope's
    /// drops are synthesized, retract the env-drop obligation of every move-closure binding this
    /// scope uses as anything other than the *direct callee of a call* -- the one value use that
    /// provably keeps the closure in place. Capture by a nested lambda counts as an escape too.
    ///
    /// Owned captures need no such fence: an owning closure is affine, so an escape
    /// *is* a move and the move tracker already suppresses the defining scope's teardown. The
    /// obligation travels with the value and fires wherever it really dies.
    pub(super) fn retract_escaping_closure_env_releases(&mut self, expr: ExprId) {
        if self.shared_closure_captures.is_empty() || !self.drop_elaboration_active() {
            return;
        }
        self.scan_closure_escapes(expr);
    }

    fn retract_escaped_closure(&mut self, name: NameId) {
        self.shared_closure_captures.remove(&name);
    }

    fn scan_closure_escapes(&mut self, expr: ExprId) {
        if self.shared_closure_captures.is_empty() {
            return;
        }
        match self.expr_of(expr).as_ref() {
            Expr::Variable(path) => {
                // A bare value use (return tail, `:=` rhs, non-callee argument, …) escapes.
                if let Some(Origin::Local(name)) = self.path_origin(*path) {
                    self.retract_escaped_closure(name);
                }
            },
            Expr::Call(call) => {
                let call = call.clone();
                // `m arg…` (a direct variable callee) does not escape `m`; a compound callee is scanned.
                if !matches!(self.expr_of(call.function).as_ref(), Expr::Variable(_)) {
                    self.scan_closure_escapes(call.function);
                }
                for arg in &call.arguments {
                    self.scan_closure_escapes(arg.expr);
                }
            },
            Expr::Lambda(_) => {
                // Capture by a nested lambda copies the closure into another env -- an escape.
                let captured: Vec<NameId> = self
                    .shared_closure_captures
                    .keys()
                    .copied()
                    .filter(|k| self.lambda_captures_name(expr, *k))
                    .collect();
                for name in captured {
                    self.retract_escaped_closure(name);
                }
            },
            Expr::Sequence(items) => {
                let items = items.clone();
                for item in &items {
                    self.scan_closure_escapes(item.expr);
                }
            },
            Expr::Definition(def) => {
                let rhs = def.rhs;
                self.scan_closure_escapes(rhs);
            },
            Expr::MemberAccess(access) => {
                let object = access.object;
                self.scan_closure_escapes(object);
            },
            Expr::If(if_) => {
                let (condition, then, else_) = (if_.condition, if_.then, if_.else_);
                self.scan_closure_escapes(condition);
                self.scan_closure_escapes(then);
                if let Some(else_) = else_ {
                    self.scan_closure_escapes(else_);
                }
            },
            Expr::Match(match_) => {
                let match_ = match_.clone();
                self.scan_closure_escapes(match_.expression);
                for (_, branch) in &match_.cases {
                    self.scan_closure_escapes(*branch);
                }
            },
            Expr::Handle(handle) => {
                let handle = handle.clone();
                self.scan_closure_escapes(handle.expression);
                for (_, branch) in &handle.cases {
                    self.scan_closure_escapes(*branch);
                }
            },
            Expr::Reference(reference) => {
                let rhs = reference.rhs;
                self.scan_closure_escapes(rhs);
            },
            Expr::TypeAnnotation(annotation) => {
                let lhs = annotation.lhs;
                self.scan_closure_escapes(lhs);
            },
            Expr::Constructor(constructor) => {
                let fields: Vec<ExprId> = constructor.fields.iter().map(|(_, e)| *e).collect();
                for field in fields {
                    self.scan_closure_escapes(field);
                }
            },
            Expr::While(while_) => {
                let (condition, body) = (while_.condition, while_.body);
                self.scan_closure_escapes(condition);
                self.scan_closure_escapes(body);
            },
            Expr::For(for_) => {
                let (start, end, body) = (for_.start, for_.end, for_.body);
                self.scan_closure_escapes(start);
                self.scan_closure_escapes(end);
                self.scan_closure_escapes(body);
            },
            Expr::Return(return_) => {
                let expression = return_.expression;
                self.scan_closure_escapes(expression);
            },
            Expr::Assignment(assignment) => {
                let (lhs, rhs, op) = (assignment.lhs, assignment.rhs, assignment.op.clone());
                self.scan_closure_escapes(lhs);
                self.scan_closure_escapes(rhs);
                if let Some((_, op)) = op {
                    self.scan_closure_escapes(op);
                }
            },
            Expr::ArrayLiteral(elements) => {
                let elements = elements.clone();
                for element in elements {
                    self.scan_closure_escapes(element);
                }
            },
            // `Is`/`Do`/`Loop`/`InterpolatedString` are desugared before type inference (the infer
            // arms `unreachable!` on them), so they never reach this CST walk.
            Expr::Error
            | Expr::Literal(_)
            | Expr::Extern(_)
            | Expr::Break
            | Expr::Continue
            | Expr::Quoted(_)
            | Expr::Is(_)
            | Expr::Do(_)
            | Expr::Loop(_)
            | Expr::InterpolatedString(_) => {},
        }
    }

    /// Derive the drop for a (possibly partially-moved) sum-typed place: an exhaustive
    /// synthesized `match` binding each variant's payloads to fresh names and dropping the
    /// ones still owned. When `tracker` is given (residual drops), payloads whose
    /// variant-qualified path is moved are left alone -- sound because `place.Some#0` can only be
    /// moved on executions where the tag was `Some`. Returns `None` when no variant payload needs
    /// dropping (pruning).
    fn synthesize_sum_drop(
        &mut self, place: &MovePath, typ: &Type, tracker: Option<&MoveTracker>, location: &Location,
    ) -> Option<ExprId> {
        let typ = self.follow_type(typ).clone();
        let (type_name, args) = match &typ {
            Type::UserDefined(Origin::TopLevelDefinition(name)) => (*name, None),
            Type::Application(constructor, args) => match self.follow_type(constructor) {
                Type::UserDefined(Origin::TopLevelDefinition(name)) => (*name, Some(args.clone())),
                _ => return None,
            },
            _ => return None,
        };

        // Variant NameIds come from the raw item (for constructor path origins); payload
        // types come from `type_body` (generics substituted). Same declaration order.
        let (raw_item, _) = crate::incremental::GetItemRaw(type_name.top_level_item).get(self.compiler);
        let cst::TopLevelItemKind::TypeDefinition(type_definition) = &raw_item.kind else { return None };
        let cst::TypeDefinitionBody::Enum(raw_variants) = &type_definition.body else { return None };
        let raw_variant_names: Vec<NameId> = raw_variants.iter().map(|(name, _)| *name).collect();

        let body = type_name.top_level_item.type_body(args.as_deref().map(|a| &a[..]), self.compiler);
        let super::type_body::TypeBody::Sum(variants) = body else { return None };
        if raw_variant_names.len() != variants.len() {
            return None;
        }

        let mut any_drops = false;
        let mut cases = Vec::new();
        for (variant_name_id, (variant_name, payload_types)) in raw_variant_names.iter().zip(&variants) {
            let mut argument_patterns = Vec::new();
            let mut drops = Vec::new();
            for (index, payload_type) in payload_types.iter().enumerate() {
                let (_, payload_binding) = self.fresh_variable("drop_payload", payload_type.clone(), location.clone());
                self.name_types.insert(payload_binding, payload_type.clone());
                let pattern = self.push_pattern(cst::Pattern::Variable(payload_binding), location.clone());
                argument_patterns.push(pattern);

                // Residual filter: this payload's qualified place may already be moved.
                let qualified = MovePath::field(place.clone(), format!("{variant_name}#{index}"));
                if let Some(tracker) = tracker
                    && tracker.is_moved(&qualified).is_some()
                {
                    continue;
                }
                let payload_place = MovePath::Variable(payload_binding);
                if let Some(drop) = self.try_synthesize_drop_for_place(&payload_place, payload_type, location) {
                    drops.push(drop);
                }
            }

            let constructor_path = self.push_path(
                cst::Path { components: vec![(variant_name.as_ref().clone(), location.clone())] },
                self.next_type_variable(),
                location.clone(),
            );
            let variant_top_level_name = TopLevelName::new(type_name.top_level_item, *variant_name_id);
            self.current_extended_context_mut()
                .insert_path_origin(constructor_path, Origin::TopLevelDefinition(variant_top_level_name));
            let pattern = self.push_pattern(cst::Pattern::Constructor(constructor_path, argument_patterns), location.clone());

            let body = if drops.is_empty() {
                self.push_expr(Expr::Literal(cst::Literal::Unit), Type::UNIT, location.clone())
            } else {
                any_drops = true;
                let items = drops.into_iter().map(|expr| cst::SequenceItem { comments: Vec::new(), expr }).collect();
                self.push_expr(Expr::Sequence(items), Type::UNIT, location.clone())
            };
            cases.push((pattern, body));
        }

        if !any_drops {
            return None;
        }

        let scrutinee = self.synthesize_place_expr(place, &typ, location);
        let match_expr =
            self.push_expr(Expr::Match(cst::Match { expression: scrutinee, cases }), Type::UNIT, location.clone());

        let old_synthesizing = std::mem::replace(&mut self.synthesizing_drops, true);
        let old_check = std::mem::replace(&mut self.suppress_move_check, true);
        let old_record = std::mem::replace(&mut self.suppress_move_record, true);
        self.check_expr(match_expr, &Type::UNIT, TypeErrorKind::General);
        self.suppress_move_record = old_record;
        self.suppress_move_check = old_check;
        self.synthesizing_drops = old_synthesizing;

        Some(match_expr)
    }

    /// Tracker-aware drop synthesis for one place: whole drop if fully owned, residual drop
    /// (unmoved components only) if partially moved, nothing if moved. Every skip leaks,
    /// never double-frees.
    pub(super) fn synthesize_partial_drop(
        &mut self, place: &MovePath, typ: &Type, tracker: &MoveTracker, location: &Location,
    ) -> Option<ExprId> {
        // A shared handle is reference-counted and unconditionally Copy: its payloads cannot be
        // moved out (projecting them copies the pointer + retains), so its own refcount must be
        // released on every binding death regardless of how the move tracker views projections out
        // of it. Matching a node and rebuilding it (`Tree _ l x r -> Tree B l x r`) reads as child
        // moves, which would otherwise route to the partial-move path below and skip the release
        // (Copy → `None`) -- leaking the handle. Bypass the move gates and always emit `release_T`.
        // Any escaping tail use is balanced by the escape retain, so this never double-frees.
        {
            let followed = self.follow_type(typ).clone();
            if self.is_shared_user_defined(&followed) {
                return self.try_synthesize_drop_for_place(place, &followed, location);
            }
        }
        if tracker.is_moved(place).is_some() {
            return None;
        }
        if tracker.has_child_moved(place).is_none() {
            return self.try_synthesize_drop_for_place(place, typ, location);
        }

        // Partially moved: drop the unmoved remainder.
        if self.captured_names.contains(&place.root_variable()) {
            return None;
        }
        let typ = self.follow_type(typ).clone();
        if typ == Type::ERROR || !typ.free_vars(&self.bindings).is_empty() || self.type_is_copy(&typ) {
            return None;
        }
        // A type with its own Drop impl cannot run it on a partially-moved value (Rust
        // forbids partial moves out of such types; a matching checker rule is future work).
        if self.type_has_drop_impl(&typ) {
            return None;
        }
        if self.drop_expansion_depth >= 16 {
            return None;
        }
        self.drop_expansion_depth += 1;

        let fields = self.get_field_types(&typ, None);
        let result = if fields.is_empty() {
            self.synthesize_sum_drop(place, &typ, Some(tracker), location)
        } else {
            let mut ordered: Vec<(String, Type, u32)> =
                fields.into_iter().map(|(name, (typ, index))| (name.to_string(), typ, index)).collect();
            ordered.sort_unstable_by_key(|(_, _, index)| *index);
            let mut drops = Vec::new();
            for (field_name, field_type, _) in ordered {
                let field_place = MovePath::field(place.clone(), field_name);
                if let Some(drop) = self.synthesize_partial_drop(&field_place, &field_type, tracker, location) {
                    drops.push(drop);
                }
            }
            if drops.is_empty() {
                None
            } else {
                let items = drops.into_iter().map(|expr| cst::SequenceItem { comments: Vec::new(), expr }).collect();
                Some(self.push_expr(Expr::Sequence(items), Type::UNIT, location.clone()))
            }
        };

        self.drop_expansion_depth -= 1;
        result
    }

    /// Reject returning a reference derived from an owned local -- auto-drop frees the referent at
    /// function exit, so the escaped reference dangles. Parameter-derived references (roots of
    /// reference/pointer type) stay legal. Gaps: References laundered through intermediate
    /// ref-typed bindings, stored into escaping structs via bindings, or captured by escaping
    /// closures.
    ///
    /// Walk a function's return/tail expression and mark every tail-position shared-handle *place*
    /// (variable / field access) for an escape retain.  Recurses through the value-producing tails
    /// of `if`/`match`/sequences so that `else tree` and `other -> other` are reached, but stops at
    /// fresh results (calls, constructors), which are already +1 and moved out.
    pub(super) fn mark_tail_escape_retains(&mut self, expr: ExprId) {
        if !self.drop_elaboration_active() {
            return;
        }
        let cst_expr = match self.current_extended_context().extended_expr(expr) {
            Some(expr) => expr.clone(),
            None => self.current_context()[expr].clone(),
        };
        match cst_expr {
            cst::Expr::Variable(_) | cst::Expr::MemberAccess(_) => {
                if self.expr_types.get(&expr).cloned().is_some_and(|typ| {
                    self.is_shared_user_defined(&typ) || self.is_heap_env_closure(&typ)
                }) {
                    self.current_extended_context_mut().mark_escape_retain(expr);
                }
            },
            cst::Expr::Sequence(items) => {
                if let Some(last) = items.last() {
                    self.mark_tail_escape_retains(last.expr);
                }
            },
            cst::Expr::If(if_) => {
                self.mark_tail_escape_retains(if_.then);
                if let Some(else_) = if_.else_ {
                    self.mark_tail_escape_retains(else_);
                }
            },
            cst::Expr::Match(match_) => {
                for (_, body) in &match_.cases {
                    self.mark_tail_escape_retains(*body);
                }
            },
            cst::Expr::TypeAnnotation(annotation) => self.mark_tail_escape_retains(annotation.lhs),
            _ => {},
        }
    }

    /// Auto-drop escape check: The value leaving this function must carry no reference into a local
    /// it frees on return. Delegates to the origin analysis ([`Self::reject_escaping_origins`],
    /// escape check v2) -- which propagates through bindings, constructors, and closure
    /// captures.
    pub(super) fn check_reference_escape(&mut self, returned: ExprId) {
        if !self.drop_elaboration_active() {
            return;
        }
        self.reject_escaping_origins(returned);
    }

    /// True if `needle` occurs anywhere within `haystack` (followed, structural equality).
    pub(super) fn type_occurs_in(&self, needle: &Type, haystack: &Type) -> bool {
        let haystack = self.follow_type(haystack);
        if self.follow_type(needle) == haystack {
            return true;
        }
        match haystack {
            Type::Application(constructor, args) => {
                self.type_occurs_in(needle, constructor) || args.iter().any(|arg| self.type_occurs_in(needle, arg))
            },
            Type::Tuple(elements) => elements.iter().any(|element| self.type_occurs_in(needle, element)),
            _ => false,
        }
    }

    pub(super) fn type_contains_reference(&self, typ: &Type) -> bool {
        let typ = self.follow_type(typ);
        match typ {
            Type::Application(constructor, args) => {
                constructor.reference_constructor(&self.bindings).is_some()
                    || args.iter().any(|arg| self.type_contains_reference(arg))
            },
            Type::Tuple(elements) => elements.iter().any(|element| self.type_contains_reference(element)),
            _ => false,
        }
    }

    /// True if the reference target is a place owned by the current function (an owned
    /// local or by-value parameter) or an rvalue temporary -- anything auto-drop frees on
    /// exit. References rooted at reference/pointer-typed bindings or at globals point at
    /// storage that outlives the call.
    pub(super) fn reference_target_is_function_local(&self, target: ExprId) -> bool {
        match self.expr_of(target).as_ref() {
            Expr::Variable(path) => match self.path_origin(*path) {
                Some(Origin::Local(name)) => {
                    // Captured outer locals are not dropped when *this* function returns;
                    // only names owned by the current function's own scopes count.
                    if !self.name_is_local_to_current_function(name) {
                        return false;
                    }
                    let Some(typ) = self.name_types.get(&name) else { return false };
                    typ.reference_element(&self.bindings).is_none() && typ.pointer_element(&self.bindings).is_none()
                },
                // Globals and builtins outlive everything; unresolved paths already errored.
                _ => false,
            },
            Expr::MemberAccess(access) => self.reference_target_is_function_local(access.object),
            Expr::TypeAnnotation(annotation) => self.reference_target_is_function_local(annotation.lhs),
            // An rvalue temporary dies at the end of the statement -- unless it diverges
            // (`fail ()` auto-ref'd to meet a reference-typed branch never comes back).
            _ => {
                let diverges =
                    self.expr_types.get(&target).is_some_and(|typ| self.diverges(&typ.clone()));
                !diverges
            },
        }
    }

    /// True if `name` is an owned local registered anywhere within the innermost enclosing
    /// lambda (block scopes may already be popped by the time the body-end escape check
    /// runs, so this uses the per-function persistent set). Names owned by outer lambdas
    /// are captured outers and are not dropped when *this* function returns.
    pub(super) fn name_is_local_to_current_function(&self, name: NameId) -> bool {
        self.function_local_names.last().is_some_and(|locals| locals.contains(&name))
    }

    /// True when `name` is an owned local of *some* enclosing function on the scope stack -- i.e. a
    /// captured outer local (as opposed to a global or a through-pointer parameter).  Used only
    /// inside a handler-scoped lambda, where such a capture provably outlives the lambda.
    fn name_is_local_to_any_enclosing_function(&self, name: NameId) -> bool {
        self.function_local_names.iter().any(|locals| locals.contains(&name))
    }

    /// Reject a user `Drop` impl whose target is a `shared` type: shared handles are
    /// Copy and never tracked, so there is no coherent point to run the impl. Only checked
    /// under `--auto-drop` (the whole Drop-semantics package).
    pub(super) fn reject_shared_drop_impl(&mut self, impl_type: &Type, pattern: PatternId) {
        if !self.auto_drop {
            return;
        }
        let mut typ = self.follow_type(impl_type).clone();
        // Impls with implicit constraints are functions returning the ability type.
        if let Type::Function(function) = &typ {
            typ = self.follow_type(&function.return_type).clone();
        }
        let Type::Application(constructor, args) = &typ else { return };
        let drop_type_name = self.get_drop_type_name();
        let is_drop = matches!(
            self.follow_type(constructor),
            Type::UserDefined(Origin::TopLevelDefinition(name)) if *name == drop_type_name
        );
        if is_drop
            && let Some(arg) = args.first()
            && self.is_shared_user_defined(arg)
        {
            let location = self.current_context().pattern_location(pattern).clone();
            self.compiler.accumulate(crate::diagnostics::Diagnostic::DropImplForSharedType { location });
        }
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

        // Global impls visible from the current item's file. A candidate whose own
        // `{Drop x}` constraints cannot resolve does not count (e.g. `drop_maybe {Drop t}`
        // must not claim `Maybe NoDropStruct` -- the structural fallback handles that
        // payload; picking the impl would fail resolution loudly instead).
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
                    && let Ok(bindings) = self.try_unify(&f.return_type, &drop_of_t)
                {
                    let f = f.clone();
                    if self.drop_impl_constraints_hold(&f, bindings) {
                        found = true;
                        return true;
                    }
                }
                false
            });
            if found {
                return true;
            }
        }

        false
    }

    /// Check that a candidate Drop impl's own `{Drop x}` constraints can resolve under the
    /// unification `bindings` from matching its return type: through another impl (checked
    /// recursively, depth-guarded) or an in-scope `{Drop v}` capability for bare variables.
    /// Non-Drop constraints are assumed satisfiable, mirroring the Copy-side check.
    fn drop_impl_constraints_hold(
        &mut self, function: &super::types::FunctionType, bindings: super::types::TypeBindings,
    ) -> bool {
        if self.copy_check_depth >= 8 {
            return false;
        }
        self.copy_check_depth += 1;
        let mut merged = self.bindings.clone();
        merged.extend(bindings);
        let drop_name = self.get_drop_type_name();
        let mut holds = true;
        for parameter in function.parameters.iter().filter(|parameter| parameter.is_implicit) {
            let constraint = parameter.typ.follow_all(&merged);
            let Type::Application(constructor, args) = &constraint else { continue };
            let is_drop_constraint = matches!(
                constructor.follow(&merged),
                Type::UserDefined(Origin::TopLevelDefinition(name)) if *name == drop_name
            );
            if !is_drop_constraint {
                continue;
            }
            let Some(arg) = args.first() else { continue };
            let arg = arg.follow_all(&merged);
            let satisfiable = match &arg {
                // A numeric-literal var defaults to I32/F64 by item end; the prelude's
                // Drop impls cover the defaults, so the constraint will resolve.
                Type::Variable(id) if self.is_literal_variable(*id) => true,
                Type::Variable(_) | Type::Generic(_) => self.constraint_in_scope_for_generic(&arg, drop_name),
                _ => self.type_has_drop_impl(&arg),
            };
            if !satisfiable {
                holds = false;
                break;
            }
        }
        self.copy_check_depth -= 1;
        holds
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
