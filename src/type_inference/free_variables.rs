use std::{collections::BTreeSet, sync::Arc};

use rustc_hash::FxHashSet;

use crate::{
    name_resolution::Origin,
    parser::{
        cst,
        ids::{ExprId, NameId, PathId, PatternId},
    },
    type_inference::{
        TypeChecker,
        errors::{Locateable, TypeErrorKind},
        types::{PrimitiveType, Type, TypeBindings},
    },
};

/// True if `typ` resolves to a pointer-shaped closure env: either the raw `Pointer` primitive or
/// `Ptr T`. Lambdas filling such a slot capture their env onto the heap and store the pointer
/// instead of trying to unify the captures' tuple type against the slot's `Ptr T`.
pub(super) fn is_pointer_env(typ: &Type, bindings: &TypeBindings) -> bool {
    match typ.follow(bindings) {
        Type::Primitive(PrimitiveType::Pointer) => true,
        Type::Application(constructor, _) => {
            matches!(constructor.follow(bindings), Type::Primitive(PrimitiveType::Pointer))
        },
        _ => false,
    }
}

impl TypeChecker<'_, '_> {
    /// Finds the environment type for the given lambda. This involves finding the free variables
    /// within the lambda. This will unify the given `expected_environment_type` with the actual
    /// environment type found but will not actually perform closure conversion. Closure conversion
    /// is instead done while building the initial `Mir`.
    pub(super) fn check_for_closure(
        &mut self, id: ExprId, expected_environment_type: &Type, self_name: Option<NameId>, is_move: bool,
    ) {
        let mut context = FreeVars::default();
        if let Some(name) = self_name {
            context.defined_in_fn.insert(name);
        }
        context.find_free_variables(id, self);

        // A bare-`Pointer` env (capability / ability-method dictionary) has no capture tuple to
        // give `IMM` slots to -- `make_env_type_with_names` is never called for it -- so nothing is
        // borrowed there and the MIR builder must not materialize. (Those envs carry only borrowed
        // dictionaries anyway, which are `Copy`.)
        let mut borrowed = FxHashSet::default();
        if !is_pointer_env(expected_environment_type, &self.bindings) {
            borrowed = self.borrowed_capture_set(&context.free_vars, is_move);
            let env_type = make_env_type_with_names(&context.free_vars, self, is_move, &borrowed);
            self.unify(&env_type, expected_environment_type, TypeErrorKind::ClosureEnv, id);
        }

        if !context.free_vars.is_empty() {
            if is_move {
                self.current_extended_context_mut().mark_move_closure(id);
            }
            // A borrowed capture is an `IMM` ref into the owner's storage: the env does not alias
            // the value, so the owner is its sole owner and must drop it normally. Restore it to
            // its scope by removing it from `captured_names`.
            //
            // Sound because a closure capturing `name` is necessarily defined *after* it, and scope
            // exit drops in reverse-definition order -- so the borrowing closure always dies before
            // its owner. A closure that escapes by return while borrowing is a compile error.
            //
            // Done here, not at the `record_captured_names` call site: this check may be **deferred**
            // (`push_deferred_closure_check`) until the enclosing scope resolves its implicits, so
            // this is the only point where `borrowed` is known. Both paths run after
            // `record_captured_names`, so the removal sticks. Handler-scoped lambdas never entered
            // `captured_names`, making the removal a no-op for them.
            for name in &borrowed {
                self.captured_names.remove(name);
            }
            self.current_extended_context_mut().insert_borrowed_captures(id, borrowed);
            self.current_extended_context_mut().insert_closure_environment(id, context.free_vars);
        }
    }

    /// hich captures this closure holds **by reference** (`IMM` in the env) instead of by
    /// value. Exactly the captures the escape check treats as `capture_borrow`
    /// (`origins.rs::lambda_origin` + the `Copy` filter in `reject_escaping_origins`) -- keeping the
    /// two in step is the invariant that makes borrow-in-env sound: every borrow that could escape
    /// is a borrow the escape check rejects.
    ///
    /// - `move` captures are owned by the env, never borrowed.
    /// - `var` captures are already `MUT` refs into the owner's slot.
    /// - reference-typed captures point elsewhere; re-wrapping would double-indirect.
    /// - **`Copy`** captures are bit-copies that cannot dangle and need no owner: primitives,
    ///   function values, ability dictionaries, and `shared` handles.
    ///
    /// `type_is_copy` fails *safe* here: an in-flight unification variable answers "Copy", so an
    /// unresolved capture stays by-value (the pre-existing leak) rather than becoming a wrong
    /// borrow. Only confidently-non-`Copy` captures are borrowed.
    fn borrowed_capture_set(&mut self, free_vars: &BTreeSet<NameId>, is_move: bool) -> FxHashSet<NameId> {
        let mut borrowed = FxHashSet::default();
        if is_move || !self.drop_elaboration_active() {
            return borrowed;
        }
        for name in free_vars {
            if self.mutable_definitions.contains(name) || self.name_is_reference_typed(*name) {
                continue;
            }
            let typ = self.name_types[name].clone();
            if !self.type_is_copy(&typ) {
                borrowed.insert(*name);
            }
        }
        borrowed
    }

    /// Auto-drop: record every free variable of the lambda at `id` as captured. Captured
    /// names are never auto-dropped by their owning scope (see `drop_elaboration.rs`):
    /// closures capture by reference and may outlive the scope, so dropping the referent
    /// would dangle the closure. Skipping only leaks, for now.
    pub(super) fn record_captured_names(&mut self, id: ExprId) {
        let mut context = FreeVars::default();
        context.find_free_variables(id, self);
        self.captured_names.extend(context.free_vars.iter().copied());
    }

    /// True when the lambda at `id` captures `name` as a free variable.
    pub(super) fn lambda_captures_name(&self, id: ExprId, name: NameId) -> bool {
        let mut context = FreeVars::default();
        context.find_free_variables(id, self);
        context.free_vars.contains(&name)
    }

    pub(super) fn record_move_captures(&mut self, id: ExprId, self_name: Option<NameId>) {
        let mut context = FreeVars::default();
        if let Some(name) = self_name {
            context.defined_in_fn.insert(name);
        }
        context.find_free_variables(id, self);

        let location = id.locate(self);
        let mut owned_captures = Vec::new();
        for name in &context.free_vars {
            let typ = self.name_types[name].clone();
            if !self.type_is_copy(&typ) {
                // Capturing a binding moves the place it denotes, as a direct use would.
                let move_path = self.binding_place(*name);
                self.move_tracker.record_move(move_path, location.clone());
                // The env is this capture's sole owner, so record it as an env-drop obligation of
                // the closure's binding. `var` captures are excluded: a `move` closure snapshots
                // them by value (`pack_closure_environment` Derefs the current value) while the
                // outer slot stays live, so dropping both would double-free -- `var` env drops are
                // out of scope here (tasks 06/10). Shared handles and function values are Copy, so
                // they never reach this branch (05 owns shared captures; heap-env closure captures
                // are 04/05).
                if !self.mutable_definitions.contains(name) {
                    owned_captures.push(*name);
                }
            }
        }
        // A `move`-captured owned value is recorded moved above, so the owner's scope-exit drop is
        // already suppressed by the move tracker (not by the blanket). Remove it from
        // `captured_names` so the blanket's remaining population is meaningful -- exactly the
        // escaping non-`move` owned captures task 10 turns into borrows.  Nothing changes at the
        // owner (a moved value has no owner-side drop either way).
        for capture in &owned_captures {
            self.captured_names.remove(capture);
        }
    }

    /// A `shared` handle captured **by value** into a closure env is a bit-copy the owner still
    /// holds too (shared handles are `Copy`, so `record_move_captures` never sees them). Rather
    /// than pick a unique owner, take a reference: the pack-time `RcRetain`
    /// (`pack_closure_environment`) gives the env its own count, so:
    ///   - the capture is **restored** to the owner's scope -- removed from `captured_names` so the
    ///     owner's scope-exit `release_T` fires again (safe now: the retain kept the count above what
    ///     the closure still needs), and
    ///   - for a **bound** closure, the balancing env-side release is recorded as an obligation of the
    ///     binding (`shared_closure_captures`), fired when the closure value dies un-escaped
    ///     (`try_synthesize_drop_for_place`).
    ///
    /// `var` captures are excluded. Anonymous closures get only the owner restore: their retain has
    /// no binding death to balance it and leaks by one count -- the leak-not-UAF fallback.  Runs
    /// after `record_captured_names` so the removal sticks; auto-drop only.
    pub(super) fn record_shared_captures(&mut self, id: ExprId, self_name: Option<NameId>) {
        let mut context = FreeVars::default();
        if let Some(name) = self_name {
            context.defined_in_fn.insert(name);
        }
        context.find_free_variables(id, self);

        let mut shared_captures = Vec::new();
        for name in &context.free_vars {
            let typ = self.name_types[name].clone();
            if self.is_shared_user_defined(&typ) && !self.mutable_definitions.contains(name) {
                shared_captures.push(*name);
            }
        }
        if shared_captures.is_empty() {
            return;
        }
        // Owner restore: the pack-time retain balances the owner's release, so let the owner drop
        // its copy again (both owner and env now hold a real count).
        for name in &shared_captures {
            self.captured_names.remove(name);
        }
        // Only a *bound* closure has a scope-exit death to release the env's reference at.
        if let Some(binding) = self_name {
            self.shared_closure_captures.insert(binding, shared_captures);
        }
    }

    /// True if the given branch of a Handle expression references its `resume` variable.
    pub(super) fn handler_branch_uses_resume(&self, resume_name: NameId, branch: ExprId) -> bool {
        let cst::Expr::Lambda(lambda) = &self.current_extended_context()[branch] else { unreachable!() };
        let mut context = FreeVars::default();
        context.find_free_variables(lambda.body, self);
        context.free_vars.contains(&resume_name)
    }
}

#[derive(Default)]
struct FreeVars {
    /// The free variables found
    free_vars: BTreeSet<NameId>,

    // We don't care about different scopes within the function
    defined_in_fn: FxHashSet<NameId>,
}

impl FreeVars {
    fn find_free_variables(&mut self, expr: ExprId, checker: &TypeChecker) {
        self.find_free_variables_inner(expr, checker);

        // Auto-drop: synthesized drop calls live in side tables keyed by this expression,
        // not in the expression tree itself -- but they are real code lowered at this
        // point, and they may reference `{Drop t}` capability parameters (a capture when
        // inside a closure). Walk them too -- after the expression itself, since e.g. a
        // block's end-of-scope drops refer to locals the block declares.
        let context = checker.current_extended_context();
        let table_drops: Vec<ExprId> = [
            context.post_expr_drops(expr),
            context.pre_exit_drops(expr),
            context.implicit_else_drops(expr),
        ]
        .into_iter()
        .flatten()
        .flatten()
        .copied()
        .collect();
        for drop_call in table_drops {
            self.find_free_variables(drop_call, checker);
        }
    }

    fn find_free_variables_inner(&mut self, expr: ExprId, checker: &TypeChecker) {
        match &checker.current_extended_context()[expr] {
            cst::Expr::Error => (),
            cst::Expr::Literal(_) => (),
            cst::Expr::Extern(_) => (),
            cst::Expr::Variable(path) => self.find_free_variable(*path, checker),
            cst::Expr::Sequence(items) => {
                for item in items {
                    self.find_free_variables(item.expr, checker);
                }
            },
            cst::Expr::Definition(definition) => {
                self.declare_pattern(definition.pattern, checker);
                self.find_free_variables(definition.rhs, checker);
            },
            cst::Expr::MemberAccess(access) => self.find_free_variables(access.object, checker),
            cst::Expr::Call(call) => {
                self.find_free_variables(call.function, checker);
                for argument in call.arguments.iter() {
                    self.find_free_variables(argument.expr, checker);
                }
            },
            cst::Expr::Lambda(lambda) => {
                for parameter in lambda.parameters.iter() {
                    self.declare_pattern(parameter.pattern, checker);
                }
                self.find_free_variables(lambda.body, checker);
            },
            cst::Expr::If(if_) => {
                self.find_free_variables(if_.condition, checker);
                self.find_free_variables(if_.then, checker);
                if let Some(else_) = if_.else_ {
                    self.find_free_variables(else_, checker);
                }
            },
            cst::Expr::Match(match_) => {
                self.find_free_variables(match_.expression, checker);
                for (pattern, branch) in match_.cases.iter() {
                    self.declare_pattern(*pattern, checker);
                    self.find_free_variables(*branch, checker);
                }
            },
            cst::Expr::Is(_) => unreachable!("Expr::Is should be desugared during GetItem"),
            cst::Expr::Do(_) => unreachable!("Expr::Do should be desugared during GetItem"),
            cst::Expr::Handle(handle) => {
                // The handler name is declared by the `handle` itself, not by any
                // surrounding scope. Treat it as defined-in-fn so references to it
                // inside the wrapped body lambda aren't reported as captures of the
                // surrounding function.
                self.defined_in_fn.insert(handle.handler_name);
                self.find_free_variables(handle.expression, checker);
                for (pattern, branch) in handle.cases.iter() {
                    for argument in pattern.args.iter() {
                        self.declare_pattern(*argument, checker);
                    }
                    // The synthetic `resume` binding is introduced for each
                    // branch; declare it so it isn't counted as a free variable.
                    self.defined_in_fn.insert(pattern.resume_name);
                    self.find_free_variables(*branch, checker);
                }
            },
            cst::Expr::Reference(reference) => self.find_free_variables(reference.rhs, checker),
            cst::Expr::TypeAnnotation(annotation) => self.find_free_variables(annotation.lhs, checker),
            cst::Expr::Constructor(constructor) => {
                for (_name, expr) in constructor.fields.iter() {
                    self.find_free_variables(*expr, checker);
                }
            },
            cst::Expr::Loop(_) => unreachable!("Loops should be desugared before finding free variables"),
            cst::Expr::While(w) => {
                self.find_free_variables(w.condition, checker);
                self.find_free_variables(w.body, checker);
            },
            cst::Expr::For(fo) => {
                self.find_free_variables(fo.start, checker);
                self.find_free_variables(fo.end, checker);
                self.defined_in_fn.insert(fo.variable);
                self.find_free_variables(fo.body, checker);
            },
            cst::Expr::Break | cst::Expr::Continue => (),
            cst::Expr::Quoted(_) => (),
            cst::Expr::Return(return_) => self.find_free_variables(return_.expression, checker),
            cst::Expr::Assignment(assignment) => {
                self.find_free_variables(assignment.lhs, checker);
                self.find_free_variables(assignment.rhs, checker);
                if let Some((_, op_expr)) = assignment.op {
                    self.find_free_variables(op_expr, checker);
                }
            },
            cst::Expr::InterpolatedString(_) => {
                unreachable!("InterpolatedString should be desugared before finding free variables")
            },
            cst::Expr::ArrayLiteral(elements) => {
                for element in elements.clone() {
                    self.find_free_variables(element, checker);
                }
            },
        }
    }

    /// Inserts any [NameId]s of values within this [PatternId] into `self.defined_in_fn`
    fn declare_pattern(&mut self, pattern: PatternId, checker: &TypeChecker) {
        match &checker.current_extended_context()[pattern] {
            cst::Pattern::Error => (),
            cst::Pattern::Variable(name) => {
                self.defined_in_fn.insert(*name);
            },
            cst::Pattern::Literal(_) => (),
            cst::Pattern::Constructor(_, fields) => {
                for field in fields {
                    self.declare_pattern(*field, checker);
                }
            },
            cst::Pattern::TypeAnnotation(pattern, _) => self.declare_pattern(*pattern, checker),
            cst::Pattern::MethodName { type_name: _, item_name } => {
                self.defined_in_fn.insert(*item_name);
            },
            cst::Pattern::Or(alts) => {
                // Each alt binds the same names, so we only need to walk the first.
                if let Some(alt) = alts.first() {
                    self.declare_pattern(*alt, checker);
                }
            },
            cst::Pattern::Alias(name, inner) => {
                self.defined_in_fn.insert(*name);
                self.declare_pattern(*inner, checker);
            },
        }
    }

    fn find_free_variable(&mut self, path: PathId, checker: &TypeChecker) {
        if let Some(Origin::Local(name)) = checker.path_origin(path) {
            self.check_name(name);
        }
    }

    fn check_name(&mut self, name: NameId) {
        if !self.defined_in_fn.contains(&name) {
            self.free_vars.insert(name);
        }
    }
}

fn make_env_type_with_names(
    free_vars: &BTreeSet<NameId>, checker: &TypeChecker, is_move: bool, borrowed: &FxHashSet<NameId>,
) -> Type {
    let free_vars = free_vars.iter().map(|name| {
        let typ = checker.name_types[name].clone();

        // Closures:
        // - Capture mutable variables by reference (a `Mut` ref)
        // - Capture owned (non-`Copy`) immutable variables by reference too, so the owner keeps
        //   ownership and drops them. `borrowed` is the frontend's authoritative set (see
        //   `borrowed_capture_set`).
        // - Capture everything else (`Copy` values, `shared` handles) by value
        // - Capture everything by move if it is a `move` closure
        if !is_move && checker.mutable_definitions.contains(name) {
            let lifetime = checker.next_type_variable();
            Type::Application(Arc::new(Type::MUT), Arc::new(vec![lifetime, typ]))
        } else if borrowed.contains(name) {
            let lifetime = checker.next_type_variable();
            Type::Application(Arc::new(Type::IMM), Arc::new(vec![lifetime, typ]))
        } else {
            typ
        }
    });
    make_env_type(free_vars)
}

fn make_env_type(free_vars: impl ExactSizeIterator<Item = Type>) -> Type {
    if free_vars.len() == 0 {
        return Type::Primitive(PrimitiveType::NoClosureEnv);
    }
    Type::Tuple(Arc::new(free_vars.collect()))
}
