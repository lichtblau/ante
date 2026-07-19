//! Origin analysis
//!
//! For every reference-carrying expression, compute which owned places its references may point
//! into. One map ([`OriginSet`] per expression, plus a `NameId -> OriginSet` binding map so
//! propagation through `r2 = r` is a table lookup). The escape check consumes it: a tail
//! `Local` origin is a dangling return.
//!
//! This is not a borrow checker: no lifetime syntax, no user-facing types, no mutation-aliasing
//! rules. It lives in the frontend next to drop elaboration (only the frontend sees nominal types)
//! and computes in evaluation order (the binding map is filled by walking statements before their
//! uses).
//!
//! `OriginSet` is a `Vec` rather than a `SmallVec<[_; 2]>` (no smallvec dep in-tree); the
//! cost fence -- origins exist only for reference-carrying / closure exprs -- keeps them tiny anyway.

use rustc_hash::FxHashMap;

use crate::{
    diagnostics::Location,
    name_resolution::Origin as NameOrigin,
    parser::{
        cst,
        ids::{ExprId, NameId},
    },
    type_inference::{TypeChecker, affine::MovePath, types::Type},
};

/// Where a reference-carrying value's references may point.
#[derive(Clone, Debug)]
#[allow(dead_code, reason = "`Param`'s NameId and `Immortal` are consumed by later passes")]
pub(super) enum OriginKind {
    /// Into a named owned local / field of this function -- dies at return.
    Local(MovePath),
    /// Into an unnamed rvalue temporary of this function -- also dies here.
    LocalTemp,
    /// Into caller-owned storage (a parameter, or a place derived from one) - legal.
    Param(NameId),
    /// Statics, literals, fresh owned values -- never dangles.
    Immortal,
}

#[derive(Clone, Debug)]
pub(super) struct Origin {
    pub(super) kind: OriginKind,
    /// This origin is a closure borrow of an owned capture, not a reference alias. These escape
    /// like any other reference -- but only when the capture is non-`Copy`; a `Copy` capture is a
    /// bit-copy in the env and cannot dangle (`reject_escaping_origins`).
    pub(super) capture_borrow: bool,
    /// Where the reference was taken (`ref x`), preserved through propagation so the escape
    /// diagnostic points at the dangling reference itself rather than the return expression.
    pub(super) location: Option<Location>,
}

pub(super) type OriginSet = Vec<Origin>;

/// The per-expression origin map produced by one function-body walk.
type OriginMap = FxHashMap<ExprId, OriginSet>;

impl TypeChecker<'_, '_> {
    /// The pure core: walk `body` in evaluation order, returning the tail expression's origin set
    /// and the full per-expression map. No side effects -- safe for 10 to call unconditionally.
    pub(super) fn compute_origins(&self, body: ExprId) -> (OriginSet, OriginMap) {
        let mut bindings: FxHashMap<NameId, OriginSet> = FxHashMap::default();
        let mut origins: OriginMap = FxHashMap::default();
        let tail = self.origin_of(body, &mut bindings, &mut origins);
        (tail, origins)
    }

    /// Escape check. Any reference in the returned value pointing into a local this function frees
    /// on return is a dangling escape: reject it with `ReferenceEscapesScope` pointing at the
    /// reference. Propagation through bindings, constructors, and closure captures.
    ///
    /// Now a closure that escapes while borrowing a non-Copy owned capture is rejected here
    /// too. It was exempt while captures were by-value bit-copies (those leak rather than dangle).
    pub(super) fn reject_escaping_origins(&mut self, returned: ExprId) {
        let (tail, _) = self.compute_origins(returned);
        let mut reported: Vec<Location> = Vec::new();
        for origin in tail.iter() {
            if !self.origin_escapes(&origin.kind) {
                continue;
            }
            // A `Copy` capture is a bit-copy in the env, never a borrow of this frame, so it
            // cannot dangle: ability dictionaries (`{p: Print t}`), function values, `shared`
            // handles and primitives all reach here. Only a non-`Copy` owned capture is a real
            // borrow the closure would carry past its owner's death.
            if origin.capture_borrow
                && let OriginKind::Local(path) = &origin.kind
            {
                let typ = self.name_types[&path.root_variable()].clone();
                if self.type_is_copy(&typ) {
                    continue;
                }
            }
            let location = origin
                .location
                .clone()
                .unwrap_or_else(|| self.current_extended_context().expr_location(returned).clone());
            if reported.contains(&location) {
                continue; // one diagnostic per distinct escaping reference
            }
            reported.push(location.clone());
            self.compiler.accumulate(crate::diagnostics::Diagnostic::ReferenceEscapesScope { location });
        }
    }

    /// True when an origin points into storage this function frees on return.
    fn origin_escapes(&self, kind: &OriginKind) -> bool {
        match kind {
            OriginKind::Local(path) => self.name_is_local_to_current_function(path.root_variable()),
            OriginKind::LocalTemp => true,
            OriginKind::Param(_) | OriginKind::Immortal => false,
        }
    }

    /// The origin set of an expression's references (the transfer table). `{}` for values that carry
    /// no reference -- scalars, fresh owned values -- so the sets stay tiny. Records each non-empty
    /// set into `origins` (the per-ExprId map).
    fn origin_of(&self, expr: ExprId, bindings: &mut FxHashMap<NameId, OriginSet>, origins: &mut OriginMap) -> OriginSet {
        let set = match self.expr_of(expr).as_ref() {
            cst::Expr::Reference(reference) => self.reference_origin(expr, reference.rhs),
            cst::Expr::Variable(path) => match self.path_origin(*path) {
                Some(NameOrigin::Local(name)) => {
                    if let Some(recorded) = bindings.get(&name) {
                        recorded.clone() // propagation through bindings (a recorded fallback origin flows through)
                    } else if self.name_is_reference_typed(name) {
                        // A ref-typed name with no recorded origin is caller-owned (a parameter).
                        vec![self.origin(OriginKind::Param(name))]
                    } else {
                        Vec::new()
                    }
                },
                _ => Vec::new(), // globals / immortal
            },
            cst::Expr::MemberAccess(access) => self.origin_of(access.object, bindings, origins),
            cst::Expr::TypeAnnotation(annotation) => self.origin_of(annotation.lhs, bindings, origins),
            cst::Expr::Sequence(items) => {
                let items = items.clone();
                let last = items.len().saturating_sub(1);
                let mut tail = Vec::new();
                for (i, item) in items.iter().enumerate() {
                    if i == last {
                        tail = self.origin_of(item.expr, bindings, origins);
                    } else {
                        self.origin_stmt(item.expr, bindings, origins);
                    }
                }
                tail
            },
            cst::Expr::Definition(_) => {
                self.origin_stmt(expr, bindings, origins);
                Vec::new()
            },
            cst::Expr::If(if_) => {
                let (then, else_) = (if_.then, if_.else_);
                let mut set = self.origin_of(then, bindings, origins);
                if let Some(else_) = else_ {
                    set.extend(self.origin_of(else_, bindings, origins));
                }
                set
            },
            cst::Expr::Match(match_) => {
                let cases: Vec<ExprId> = match_.cases.iter().map(|(_, branch)| *branch).collect();
                cases.into_iter().flat_map(|branch| self.origin_of(branch, bindings, origins)).collect()
            },
            cst::Expr::Constructor(constructor) => {
                let fields: Vec<ExprId> = constructor.fields.iter().map(|(_, e)| *e).collect();
                fields.into_iter().flat_map(|f| self.origin_of(f, bindings, origins)).collect()
            },
            cst::Expr::Call(call) => {
                let call = call.clone();
                self.call_origin(&call, expr, bindings, origins)
            },
            cst::Expr::Lambda(_) => self.lambda_origin(expr, bindings),
            _ => Vec::new(),
        };
        if !set.is_empty() {
            origins.insert(expr, set.clone());
        }
        set
    }

    /// Record a non-tail statement's effect on the binding→origin map (the stateful step).
    fn origin_stmt(&self, expr: ExprId, bindings: &mut FxHashMap<NameId, OriginSet>, origins: &mut OriginMap) {
        match self.expr_of(expr).as_ref() {
            cst::Expr::Definition(definition) => {
                let (rhs, pattern) = (definition.rhs, definition.pattern);
                let rhs_origins = self.origin_of(rhs, bindings, origins);
                if let Some(name) = self.single_variable_pattern(pattern) {
                    bindings.insert(name, rhs_origins);
                }
            },
            // `r := s` into a reference-typed `var`: the binding may now point where either pointed
            // (may-information -- overwrite-narrowing is a later precision refinement).
            cst::Expr::Assignment(assignment) => {
                let (lhs, rhs) = (assignment.lhs, assignment.rhs);
                let rhs_origins = self.origin_of(rhs, bindings, origins);
                if let cst::Expr::Variable(path) = self.expr_of(lhs).as_ref()
                    && let Some(NameOrigin::Local(name)) = self.path_origin(*path)
                {
                    bindings.entry(name).or_default().extend(rhs_origins);
                }
            },
            cst::Expr::Sequence(items) => {
                let items = items.clone();
                for item in &items {
                    self.origin_stmt(item.expr, bindings, origins);
                }
            },
            _ => {},
        }
    }

    /// `ref`/`mut`/`imm`/`uniq <rhs>`: where the resulting reference points. `reference_expr` is the
    /// whole `ref …` expression, whose location is stamped onto `Local`/`LocalTemp` origins so the
    /// escape diagnostic points at the reference itself (preserved through propagation).
    fn reference_origin(&self, reference_expr: ExprId, rhs: ExprId) -> OriginSet {
        let location = self.current_extended_context().expr_location(reference_expr).clone();
        if let Some(path) = self.try_build_move_path(rhs) {
            let root = path.root_variable();
            // A reference to a place rooted at a reference/pointer-typed binding points where that
            // binding points -- caller-owned unless it is itself an owned local (then it dies here).
            if self.name_is_reference_typed(root) && !self.name_is_local_to_current_function(root) {
                return vec![self.origin(OriginKind::Param(root))];
            }
            if self.name_is_local_to_current_function(root) {
                return vec![self.located(OriginKind::Local(path), location)];
            }
            return vec![self.origin(OriginKind::Param(root))];
        }
        if self.reference_target_is_function_local(rhs) {
            return vec![self.located(OriginKind::LocalTemp, location)];
        }
        Vec::new()
    }

    /// Call. A constructor application stores its arguments into a nominal value whose shallow type
    /// does not read as containing a reference. Otherwise, unknown callee: the fallback heuristic -- the
    /// reference-typed arguments whose referent may contain the result's element type, propagated
    /// through bindings and marked `FALLBACK`.
    fn call_origin(
        &self, call: &cst::Call, call_expr: ExprId, bindings: &mut FxHashMap<NameId, OriginSet>, origins: &mut OriginMap,
    ) -> OriginSet {
        if self.callee_is_constructor(call.function) {
            let mut set = Vec::new();
            for argument in call.arguments.iter().filter(|a| !a.is_implicit) {
                for origin in self.origin_of(argument.expr, bindings, origins) {
                    set.push(origin);
                }
            }
            return set;
        }
        let Some(result_type) = self.expr_types.get(&call_expr).cloned() else { return Vec::new() };
        if !self.type_carries_origin(&result_type) {
            return Vec::new();
        }
        let mut result_elements = Vec::new();
        self.collect_origin_elements(&result_type, &mut result_elements);

        let mut set = Vec::new();
        for argument in call.arguments.iter().filter(|a| !a.is_implicit) {
            let Some(arg_type) = self.expr_types.get(&argument.expr).cloned() else { continue };
            let mut arg_elements = Vec::new();
            self.collect_origin_elements(&arg_type, &mut arg_elements);
            let may_alias = arg_elements.iter().any(|ae| result_elements.iter().any(|re| self.type_occurs_in(re, ae)));
            if may_alias {
                for origin in self.origin_of(argument.expr, bindings, origins) {
                    set.push(origin);
                }
            }
        }
        set
    }

    /// Like [`Self::type_contains_reference`] but also counts raw pointers. The origin analysis
    /// follows the `ref -> Ptr -> ref` laundering the `_unchecked` accessors do (`transmute` /
    /// `offset` / `ptr_to_mut`); the shallow `type_contains_reference` gate -- shared with the fallback and
    /// drop synthesis, left Ptr-blind on purpose -- stops at the `Ptr`. Used only by the
    /// unknown-callee fallback, so a pointer-returning intrinsic still forwards its argument origins.
    fn type_carries_origin(&self, typ: &Type) -> bool {
        let typ = self.follow_type(typ);
        if typ.pointer_element(&self.bindings).is_some() {
            return true;
        }
        match typ {
            Type::Application(constructor, args) => {
                constructor.reference_constructor(&self.bindings).is_some()
                    || args.iter().any(|arg| self.type_carries_origin(arg))
            },
            Type::Tuple(elements) => elements.iter().any(|element| self.type_carries_origin(element)),
            _ => false,
        }
    }

    /// Reference and pointer element types, for the fallback's may-alias test (pointer-aware; see
    /// [`Self::type_carries_origin`]).
    fn collect_origin_elements(&self, typ: &Type, out: &mut Vec<Type>) {
        let typ = self.follow_type(typ);
        if let Some(element) = typ.pointer_element(&self.bindings) {
            out.push(element);
        }
        match typ {
            Type::Application(constructor, args) => {
                if constructor.reference_constructor(&self.bindings).is_some() && args.len() == 2 {
                    out.push(args[1].clone());
                }
                for arg in args.iter() {
                    self.collect_origin_elements(arg, out);
                }
            },
            Type::Tuple(elements) => {
                for element in elements.iter() {
                    self.collect_origin_elements(element, out);
                }
            },
            _ => (),
        }
    }

    /// Closure literal: the origins of the references it captures, plus a `Local(n)` borrow for
    /// every non-`move` captured owned local `n`. A captured reference parameter points to caller
    /// storage and does not flag.
    fn lambda_origin(&self, lambda_expr: ExprId, bindings: &mut FxHashMap<NameId, OriginSet>) -> OriginSet {
        let Some(locals) = self.function_local_names.last().cloned() else { return Vec::new() };
        let is_move = matches!(self.expr_of(lambda_expr).as_ref(), cst::Expr::Lambda(lambda) if lambda.is_move);
        let mut set = Vec::new();
        for name in locals {
            if !self.lambda_captures_name(lambda_expr, name) {
                continue;
            }
            // A `let`-bound owned local (`foo = "a" ++ "b"`) records an empty origin set for its
            // binding -- it points nowhere. That is not "already handled": it is an owned capture, so
            // an empty entry must fall through to the borrow arm below exactly like a parameter's
            // absent entry. Treating `Some(&[])` as handled silently skipped every bound-then-captured
            // local (`closure_return.an`), the escape check's last laundering hole.
            if let Some(recorded) = bindings.get(&name).filter(|recorded| !recorded.is_empty()) {
                // A captured binding that already points somewhere (`r = ref local`) carries that
                // origin into the closure -- a reference alias, escape-relevant.
                for origin in recorded.clone() {
                    set.push(origin);
                }
            } else if !is_move && !self.name_is_reference_typed(name) {
                // A non-`move` closure borrows its owned captures, recorded here so the escape
                // check can reject them. Reference-typed captures point elsewhere; `move` captures are owned by
                // the env -- neither is a borrow of this function's storage.
                set.push(Origin {
                    kind: OriginKind::Local(MovePath::Variable(name)),
                    capture_borrow: true,
                    location: Some(self.current_extended_context().expr_location(lambda_expr).clone()),
                });
            }
        }
        set
    }

    fn origin(&self, kind: OriginKind) -> Origin {
        Origin { kind, capture_borrow: false, location: None }
    }

    /// An origin carrying the source location of the reference that produced it (a `Local`/`LocalTemp`
    /// created by `reference_origin`), so the escape diagnostic can point at the reference.
    fn located(&self, kind: OriginKind, location: Location) -> Origin {
        Origin { kind, capture_borrow: false, location: Some(location) }
    }

    /// True when a call's callee is a data constructor (its top-level item is a type definition).
    fn callee_is_constructor(&self, function: ExprId) -> bool {
        let function_expr = self.expr_of(function);
        let cst::Expr::Variable(path) = function_expr.as_ref() else { return false };
        let Some(NameOrigin::TopLevelDefinition(name)) = self.path_origin(*path) else { return false };
        let (item, _) = crate::incremental::GetItemRaw(name.top_level_item).get(self.compiler);
        matches!(item.kind, cst::TopLevelItemKind::TypeDefinition(_))
    }

    /// Record which explicit parameters of the reference-returning function `fn_name` have origins
    /// that flow into the return value -- its return-origin summary, keyed by item then name in
    /// `return_origin_summaries` (drained into each `IndividualTypeCheckResult` at `finish`, like
    /// the borrowed-param masks). Runs post-body while the function scope is still live, so
    /// parameters resolve to `Param` and body locals to `Local`. Non-reference returns get no entry
    /// (`call_origin` ignores them); a stored all-`false` mask means "returns a reference, but not
    /// one derived from a parameter". No codegen effect -- only the origin analysis reads it.
    pub(super) fn compute_return_origin_summary(&mut self, fn_name: NameId, lambda: &cst::Lambda, return_type: &Type) {
        if !self.drop_elaboration_active() {
            return;
        }
        let return_type = self.follow_type(return_type).clone();
        if !self.type_contains_reference(&return_type) {
            return;
        }
        let (tail, _) = self.compute_origins(lambda.body);
        let mut mask = Vec::new();
        for parameter in lambda.parameters.iter().filter(|p| !p.is_implicit) {
            let flows = self.single_variable_pattern(parameter.pattern).is_some_and(|param_name| {
                tail.iter().any(|origin| matches!(&origin.kind, OriginKind::Param(n) if *n == param_name))
            });
            mask.push(flows);
        }
        if let Some(item) = self.current_item {
            self.return_origin_summaries.entry(item).or_default().insert(fn_name, mask);
        }
    }

    pub(super) fn name_is_reference_typed(&self, name: NameId) -> bool {
        self.name_types.get(&name).is_some_and(|typ| {
            typ.reference_element(&self.bindings).is_some() || typ.pointer_element(&self.bindings).is_some()
        })
    }
}
