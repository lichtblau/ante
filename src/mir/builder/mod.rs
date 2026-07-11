//! The MIR builder will translate a single top-level item into the equivalent MIR for the item.
//! This is meant to work in parallel for each item.
//!
//! For more on the Medium-level IR (MIR) itself, see [super].
//!
//! Although the MIR may eventually be monomorphized, the initial output of this builder keeps
//! the original generics, relying on a later pass to manually either specialize each function
//! or existentialize it.
//!
//! The MIR-builder will however, perform closure conversion on any functions with closure types
//! it finds.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, LazyLock},
};

use dashmap::DashMap;
use inc_complete::DbGet;
use rustc_hash::FxHashMap;

use crate::{
    incremental::{GetItem, GetItemRaw, TypeCheck},
    iterator_extensions::mapvec,
    lexer::token::{FloatKind, Integer, IntegerKind},
    mir::{
        Block, BlockId, Definition, DefinitionId, FloatConstant, Generic, Instruction, IntConstant, Mir,
        TerminatorInstruction, Type, Value, next_definition_id,
    },
    name_resolution::Origin,
    parser::{
        cst::{self, Literal, Name, SequenceItem},
        ids::{ExprId, NameId, PathId, PatternId, TopLevelId, TopLevelName},
    },
    type_inference::{
        self,
        dependency_graph::TypeCheckResult,
        fresh_expr::ExtendedTopLevelContext,
        patterns::{Case, Constructor, DecisionTree},
        types::Type as TCType,
    },
};

mod intrinsics;
mod types;

/// Maps each TopLevelName to a unique DefinitionId
pub(crate) type SharedIdsMap = DashMap<TopLevelName, DefinitionId>;

/// A map from [TopLevelName] to [DefinitionId] shared between concurrent calls of
/// [build_initial_mir].
static NAME_IDS: LazyLock<SharedIdsMap> = LazyLock::new(DashMap::new);

/// Builds the MIR with the default shared global [SharedIdsMap].
pub(crate) fn build_initial_mir_with_shared_map<T>(compiler: &T, item_id: TopLevelId) -> Option<Mir>
where
    T: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw>,
{
    build_initial_mir(compiler, &NAME_IDS, item_id)
}

/// Convert the given item to an initial MIR representation. This may be done in parallel with all
/// other items in the program.
///
/// The initial MIR representation has no special handling of generics and requires another pass
/// afterward to reshape them into something the runtime can handle. Examples include either
/// monomorphization to specialize generics out of the code or an existential generics approach
/// which will pass around unsized values by reference.
pub(crate) fn build_initial_mir<T>(compiler: &T, ids: &SharedIdsMap, item_id: TopLevelId) -> Option<Mir>
where
    T: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw>,
{
    let types = TypeCheck(item_id).get(compiler);
    let (item, _) = GetItem(item_id).get(compiler);
    let mut context = Context::new(compiler, &types, item_id, ids);

    match &item.kind {
        cst::TopLevelItemKind::Definition(definition) => {
            context.definition(definition, true);
            Some(context.finish())
        },
        cst::TopLevelItemKind::TypeDefinition(type_definition) => {
            context.type_definition(type_definition);
            Some(context.finish())
        },
        cst::TopLevelItemKind::AbilityDefinition(_) => unreachable!("Abilities should be desguared to types"),
        cst::TopLevelItemKind::AbilityImpl(_) => unreachable!("AbilityImpls should be desugared to definitions"),
        cst::TopLevelItemKind::Comptime(_) => None,
    }
}

enum LhsKind {
    LocalVar(NameId),
    DerefCall(ExprId),
    Annotation(ExprId),
    FieldAccess(ExprId, ExprId),
    Other,
}

/// True if `typ` is a function whose return type is `Never`. Checked before `convert_type`
/// erases `Never`, since we need the divergence info to emit `Unreachable` after the call.
fn function_returns_never<'a>(mut typ: &'a TCType, bindings: &'a type_inference::types::TypeBindings) -> bool {
    use type_inference::types::PrimitiveType::Never;
    loop {
        typ = typ.follow(bindings);
        match typ {
            TCType::Forall(_, inner) => typ = inner,
            TCType::Function(f) => {
                return matches!(f.return_type.follow(bindings), TCType::Primitive(Never));
            },
            _ => return false,
        }
    }
}

/// The per-[TopLevelId] context. This pass is designed so that we can convert every top-level item
/// to MIR in parallel.
struct Context<'local, Db> {
    compiler: &'local Db,
    types: &'local TypeCheckResult,

    top_level_id: TopLevelId,

    /// The number of generics in scope
    generics_in_scope: FxHashMap<type_inference::generics::Generic, Generic>,

    current_function: Option<Definition>,
    current_block: BlockId,

    global_variables: FxHashMap<Origin, Value>,
    local_variables: FxHashMap<NameId, Value>,

    /// Names of locally-declared mutable variables (`var name = expr`).
    /// These have a StackAlloc'd pointer as their MIR value.
    mutable_locals: rustc_hash::FxHashSet<NameId>,

    /// Stack of `(continue_target, break_target)` for each enclosing `while`/`for` loop.
    /// Innermost loop is last. Swapped out across lambda boundaries.
    loop_targets: Vec<(BlockId, BlockId)>,

    finished_functions: FxHashMap<DefinitionId, Definition>,
    name_to_id: &'local SharedIdsMap,

    /// Any external items will have their name & type stored here
    external: FxHashMap<DefinitionId, super::Extern>,

    /// Cache of whether a given top-level item is an ability definition.
    ///
    /// For each Call we have to decide if it dispatches through an ability so we
    /// can potentially issue a Perform instead. This cache avoids re-issuing a
    /// GetItemRaw query for every call site.
    ability_defs: FxHashMap<TopLevelId, bool>,

    /// Position of each effect op within its ability's body. Populated as we encounter
    /// effect operations during MIR building and propagated to [Mir::preserved_op_indices]
    /// at the end so [crate::mir::effects::effect_lowering] can look up the slot of an op
    /// in the cap tuple without re-walking the ability declaration.
    effect_op_indices: FxHashMap<DefinitionId, u32>,
}

impl<'local, Db> Context<'local, Db> {
    fn new(
        compiler: &'local Db, types: &'local TypeCheckResult, top_level_id: TopLevelId,
        name_mappings: &'local SharedIdsMap,
    ) -> Self {
        Self {
            compiler,
            types,
            top_level_id,
            generics_in_scope: Default::default(),
            global_variables: FxHashMap::default(),
            local_variables: FxHashMap::default(),
            mutable_locals: Default::default(),
            loop_targets: Vec::new(),
            current_block: BlockId::ENTRY_BLOCK,
            current_function: None,
            finished_functions: Default::default(),
            name_to_id: name_mappings,
            external: Default::default(),
            ability_defs: Default::default(),
            effect_op_indices: Default::default(),
        }
    }

    /// Panics if there is no current function.
    fn current_function(&mut self) -> &mut Definition {
        self.current_function.as_mut().unwrap()
    }

    fn type_of_value(&self, value: &Value) -> Type {
        self.current_function.as_ref().unwrap().type_of_value(value, &self.external, &self.finished_functions)
    }

    /// Panics if there is no current function.
    fn current_block(&mut self) -> &mut Block {
        &mut self.current_function.as_mut().unwrap().blocks[self.current_block]
    }

    fn context(&self) -> &'local ExtendedTopLevelContext {
        &self.types.result.context
    }
}

impl<'local, Db> Context<'local, Db>
where
    Db: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw>,
{
    fn push_instruction(&mut self, instruction: Instruction, result_type: Type) -> Value {
        // Check if the block is already sealed and ignore any instructions if so
        if self.current_block().terminator.is_some() {
            return Value::Unit;
        }
        let current_block = self.current_block;
        let function = self.current_function();
        let id = function.instructions.push(instruction);
        function.instruction_result_types.push_existing(id, result_type);
        function.blocks[current_block].instructions.push(id);
        Value::InstructionResult(id)
    }

    fn push_block(&mut self, parameter_types: Vec<Type>) -> BlockId {
        self.current_function.as_mut().unwrap().blocks.push(Block::new(parameter_types))
    }

    fn push_block_no_params(&mut self) -> BlockId {
        self.push_block(Vec::new())
    }

    fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    fn push_parameter(&mut self, parameter_type: Type) {
        self.current_block().parameter_types.push(parameter_type);
    }

    fn terminate_block(&mut self, terminator: TerminatorInstruction) {
        let block = self.current_block();
        if block.terminator.is_none() {
            block.terminator = Some(terminator);
        }
    }

    fn expr_type(&self, expr: ExprId) -> Type {
        let typ = &self.types.result.maps.expr_types[&expr];
        self.convert_type(typ, None)
    }

    /// If `typ` is a `shared` user-defined type, dereference `value`, otherwise return it unchanged.
    fn deref_if_shared(&mut self, value: Value, typ: &TCType) -> Value {
        match self.shared_inner_layout_of(typ) {
            Some(inner_layout) => self.push_instruction(Instruction::Deref(value), inner_layout),
            None => value,
        }
    }

    fn define_variable(&mut self, origin: Origin, value: Value) {
        match origin {
            Origin::Local(name) => self.local_variables.insert(name, value),
            other => self.global_variables.insert(other, value),
        };
    }

    /// Retrieves the corresponding [DefinitionId] for a particular [TopLevelName].
    /// Note that this uses a shared [DashMap] internally and the resulting id will be
    /// nondeterministic across multiple compiler runs.
    fn get_definition_id(&self, name: &TopLevelName) -> DefinitionId {
        *self.name_to_id.entry(*name).or_insert_with(next_definition_id)
    }

    fn get_definition_name(&self, name: &TopLevelName) -> Name {
        // A shared type's synthesized `release_T` uses a reserved name id that has no entry in the
        // parsed name arrays; give it a stable synthetic name.
        if name.local_name_id == NameId::RELEASE_FUNCTION {
            return Arc::new("release".to_string());
        }
        let (_, context) = GetItemRaw(name.top_level_item).get(self.compiler);
        context.names[name.local_name_id].clone()
    }

    fn make_definition_value(&mut self, id: DefinitionId, name: Name, typ: Type) -> Value {
        self.reference_definition(id, name, typ);
        Value::Definition(id)
    }

    fn reference_definition(&mut self, id: DefinitionId, name: Name, typ: Type) {
        if !self.finished_functions.contains_key(&id) && self.current_function.as_ref().is_none_or(|def| def.id != id) {
            self.external.insert(id, super::Extern { name, typ });
        }
    }

    fn expression(&mut self, expr: ExprId) -> Value {
        let value = self.expression_inner(expr);

        // A shared handle place escaping via this (tail-position) expression is a new owning
        // location for the caller, so retain it -- before the scope-exit releases below, which would
        // otherwise free the returned handle/env. The frontend marks only shared and
        // heap-env-closure places.
        if self.context().is_escape_retain(expr) {
            self.emit_place_retain(expr, value);
        }

        // Auto-drop: lower any synthesized scope-exit drop calls recorded against this
        // expression (block-fallthrough and function-exit edges), after its value is computed.
        if let Some(drops) = self.context().post_expr_drops(expr) {
            for drop_call in drops.clone() {
                self.expression(drop_call);
            }
        }

        // A heap-env closure place dying on this edge releases its environment's refcount (freeing
        // the env block at zero). The frontend records the release place directly here (rather than
        // as a synthesized call -- closures have no nominal `release_T`).
        if self.context().is_closure_env_release(expr) {
            let env = self.push_instruction(Instruction::IndexTuple { tuple: value, index: 1 }, Type::POINTER);
            self.push_instruction(Instruction::ReleaseClosureEnv(env), Type::UNIT);
        }

        value
    }

    fn expression_inner(&mut self, expr: ExprId) -> Value {
        match &self.context()[expr] {
            cst::Expr::Error => unreachable!("Error expression encountered while generating boxed mir"),
            cst::Expr::Literal(literal) => self.literal(literal, expr),
            cst::Expr::Variable(path_id) => self.variable(*path_id),
            cst::Expr::Sequence(sequence) => self.sequence(sequence),
            cst::Expr::Definition(definition) => self.definition(definition, false),
            cst::Expr::MemberAccess(member_access) => self.member_access(member_access, expr),
            cst::Expr::Call(call) => self.call(call, expr),
            cst::Expr::Lambda(lambda) => self.lambda(lambda, None, None, expr, false),
            cst::Expr::If(if_) => self.if_(if_, expr),
            cst::Expr::Match(_) => self.match_(expr),
            cst::Expr::Is(_) => unreachable!("Expr::Is should be desugared during GetItem"),
            cst::Expr::Do(_) => unreachable!("Expr::Do should be desugared during GetItem"),
            cst::Expr::Handle(handle) => self.handle(handle, expr),
            cst::Expr::Reference(reference) => self.reference(reference),
            cst::Expr::TypeAnnotation(type_annotation) => self.expression(type_annotation.lhs),
            cst::Expr::Constructor(constructor) => self.constructor(constructor, expr),
            cst::Expr::Quoted(quoted) => self.quoted(quoted),
            cst::Expr::Loop(_) => unreachable!("Loops should be desugared before MIR generation"),
            cst::Expr::While(while_) => self.while_(while_),
            cst::Expr::For(for_) => self.for_(for_),
            cst::Expr::Break => self.break_(expr),
            cst::Expr::Continue => self.continue_(expr),
            cst::Expr::Return(return_) => self.return_(return_.expression),
            cst::Expr::Assignment(assignment) => self.assignment(assignment, expr),
            cst::Expr::Extern(extern_) => self.extern_(extern_, expr),
            cst::Expr::InterpolatedString(_) => {
                unreachable!("InterpolatedString should be desugared before MIR generation")
            },
            cst::Expr::ArrayLiteral(elements) => self.array_literal(elements, expr),
        }
    }

    fn array_literal(&mut self, elements: &[ExprId], expr: ExprId) -> Value {
        let result_type = self.expr_type(expr);
        let values = elements.iter().map(|e| self.expression(*e)).collect();
        self.push_instruction(Instruction::MakeArray(values), result_type)
    }

    fn literal(&mut self, literal: &Literal, expr: ExprId) -> Value {
        match literal {
            Literal::Unit => Value::Unit,
            Literal::Bool(x) => Value::Bool(*x),
            Literal::Integer(x, None) => {
                let kind = match self.expr_type(expr) {
                    Type::Primitive(crate::mir::PrimitiveType::Int(kind)) => kind,
                    _ => IntegerKind::I32,
                };
                Self::integer(*x, kind)
            },
            Literal::Integer(x, Some(kind)) => Self::integer(*x, *kind),
            Literal::Float(x, None) => {
                match self.expr_type(expr) {
                    Type::Primitive(crate::mir::PrimitiveType::Float(FloatKind::F32)) => {
                        Value::Float(FloatConstant::F32(*x))
                    },
                    // Default to F64, there are cases when the type variable may still be unbound here.
                    // Generally it means it was unused or there was an error.
                    _ => Value::Float(FloatConstant::F64(*x)),
                }
            },
            Literal::Float(x, Some(FloatKind::F32)) => Value::Float(FloatConstant::F32(*x)),
            Literal::Float(x, Some(FloatKind::F64)) => Value::Float(FloatConstant::F64(*x)),
            Literal::String(x) => self.string_literal(x),
            Literal::Char(x) => Value::Char(*x),
        }
    }

    /// Lower a string literal to the prelude String representation:
    /// `(data: Ptr Char, refcount: Ptr U32, length: U32, offset: U32)`.
    fn string_literal(&mut self, s: &str) -> Value {
        let len = s.len() as u32;

        let mut bytes = Vec::with_capacity(s.len() + 1);
        bytes.extend_from_slice(s.as_bytes());
        bytes.push(0);

        let data_ptr = self.push_instruction(Instruction::MakeBytes(bytes), Type::POINTER);
        let null_rc = self.push_instruction(Instruction::Transmute(Value::Integer(IntConstant::Usz(0))), Type::POINTER);
        let length = Value::Integer(IntConstant::U32(len));
        let offset = Value::Integer(IntConstant::U32(0));

        self.push_instruction(Instruction::MakeTuple(vec![data_ptr, null_rc, length, offset]), Type::string())
    }

    fn integer(value: Integer, kind: IntegerKind) -> Value {
        let m = value.magnitude;
        match kind {
            IntegerKind::I8 => {
                Value::Integer(IntConstant::I8(if value.negative { (m as i8).wrapping_neg() } else { m as i8 }))
            },
            IntegerKind::I16 => {
                Value::Integer(IntConstant::I16(if value.negative { (m as i16).wrapping_neg() } else { m as i16 }))
            },
            IntegerKind::I32 => {
                Value::Integer(IntConstant::I32(if value.negative { (m as i32).wrapping_neg() } else { m as i32 }))
            },
            IntegerKind::I64 => {
                Value::Integer(IntConstant::I64(if value.negative { (m as i64).wrapping_neg() } else { m as i64 }))
            },
            IntegerKind::Isz => {
                Value::Integer(IntConstant::Isz(if value.negative { (m as isize).wrapping_neg() } else { m as isize }))
            },
            IntegerKind::U8 => Value::Integer(IntConstant::U8(m as u8)),
            IntegerKind::U16 => Value::Integer(IntConstant::U16(m as u16)),
            IntegerKind::U32 => Value::Integer(IntConstant::U32(m as u32)),
            IntegerKind::U64 => Value::Integer(IntConstant::U64(m)),
            IntegerKind::Usz => Value::Integer(IntConstant::Usz(m as usize)),
        }
    }

    fn variable(&mut self, path_id: PathId) -> Value {
        // Deliberately allow us to reference variables not in the context.
        // This allows us to convert all definitions to MIR in parallel, trusting
        // that the links will work out later.
        let mut value =
            match self.context().path_origin(path_id) {
                Some(Origin::TopLevelDefinition(name)) => {
                    let id = self.get_definition_id(&name);
                    let name = self.get_definition_name(&name);
                    let typ = self.convert_path_type(path_id);
                    self.make_definition_value(id, name, typ)
                },
                Some(Origin::Local(name)) => {
                    let ptr = *self.local_variables.get(&name).unwrap_or_else(|| {
                        let function = self.current_function.as_ref().map(|f| f.name.to_string());
                        panic!(
                            "No cached variable for {} with name {name} while lowering {function:?}",
                            self.context()[path_id]
                        )
                    });
                    if self.mutable_locals.contains(&name) {
                        // Mutable locals are StackAlloc'd pointers; auto-deref to load the value.
                        let val_type = self.convert_path_type(path_id);
                        self.push_instruction(Instruction::Deref(ptr), val_type)
                    } else {
                        ptr
                    }
                },
                Some(origin @ Origin::Builtin(_)) => *self.global_variables.get(&origin).unwrap_or_else(|| {
                    panic!("No cached variable for {} with origin {origin}", self.context()[path_id])
                }),
                Some(Origin::TypeResolution) => unreachable!("Unresolved TypeResolution origin found"),
                // This is possible if there were errors during name resolution
                None => Value::Error,
            };

        // If this type was instantiated, then we need to recover the pre-instantiated
        // type and make an explicit [Instruction::Instantiate]. We cannot check this only in the
        // `Origin::TopLevelDefinition` case because local lambdas may also be polymorphic.
        // TODO: Closures may be wrapped in an Instruction result which would break this check
        if let Value::Definition(id) = value
            && let Some(bindings) = self.types.result.context.get_instantiation(path_id)
        {
            let typ = self.convert_path_type(path_id);
            let bindings = Arc::new(mapvec(bindings, |typ| self.convert_type(typ, None)));
            let instruction = Instruction::Instantiate(id, bindings);
            value = self.push_instruction(instruction, typ);
        }

        value
    }

    fn sequence(&mut self, sequence: &[SequenceItem]) -> Value {
        let mut result = Value::Unit;
        for item in sequence {
            result = self.expression(item.expr);
        }
        result
    }

    fn definition(&mut self, definition: &cst::Definition, is_global: bool) -> Value {
        let (name, name_id) = match self.try_find_name(definition.pattern) {
            Some((name, name_id)) => (name, Some(name_id)),
            None => (Arc::new("global".to_string()), None),
        };

        if is_global {
            let typ = &self.types.get_generalized(name_id.unwrap());
            self.set_generics_in_scope(typ);
        }

        let previous_state = self.is_non_function_global(definition).then(|| {
            let generic_count = self.generics_in_scope.len() as u32;
            let typ = self.convert_pattern_type(definition.pattern);
            self.start_global(name, name_id, generic_count, typ)
        });

        let value = match &self.context()[definition.rhs] {
            cst::Expr::Lambda(lambda) => {
                let name = self.try_find_name(definition.pattern).map(|(name, _)| name);
                self.lambda(lambda, name_id, name, definition.rhs, is_global)
            },
            _ => self.expression(definition.rhs),
        };

        self.bind_pattern(definition.pattern, value);
        // Binding from an existing shared handle creates a new owning location; the binding's
        // scope-exit release balances this retain. Only *real* drop-registered bindings are
        // retained -- synthesized match-variable copies (`$mv = l`, payload binds) are borrows the
        // frontend never releases, so retaining them would leak.
        if self.context().is_retain_binding(definition.rhs) {
            self.retain_if_shared_place(definition.rhs, value);
        }

        // TODO: Globals should probably never be stack allocated
        if definition.mutable {
            let mut names = Vec::new();
            self.collect_pattern_names(definition.pattern, &mut names);
            for name in names {
                let raw = *self.local_variables.get(&name).expect("var binding missing local after bind_pattern");
                let alloc = self.push_instruction(Instruction::StackAlloc(raw), Type::POINTER);
                self.local_variables.insert(name, alloc);
                self.mutable_locals.insert(name);
            }
        }

        if let Some(state) = previous_state {
            self.terminate_block(TerminatorInstruction::Result(value));
            self.end_global(state);
        }
        Value::Unit
    }

    /// True if the given definition is syntactically a global non-function variable.
    fn is_non_function_global(&self, definition: &cst::Definition) -> bool {
        self.current_function.is_none() && !matches!(self.context()[definition.rhs], cst::Expr::Lambda(_))
    }

    fn member_access(&mut self, member_access: &cst::MemberAccess, expr: ExprId) -> Value {
        let index = self.context().member_access_index(expr).unwrap_or(u32::MAX);

        // If the object has a reference or pointer type (e.g. `p: mut Point`), the MIR value is a pointer.
        // Use GetFieldPtr to produce a pointer to the field (MIR rep of e.g. `mut I32`).
        let object_expr = member_access.object;
        let object_type = self.types.result.maps.expr_types[&object_expr].follow(&self.types.bindings);
        let reference_element =
            object_type.reference_or_pointer_element(&self.types.bindings).map(|typ| self.convert_type(typ, None));

        if let Some(struct_type) = reference_element {
            let struct_ptr = self.expression(object_expr);
            self.push_instruction(Instruction::GetFieldPtr { struct_ptr, struct_type, index }, Type::POINTER)
        } else {
            let value = self.expression(object_expr);
            let tuple = self.deref_if_shared(value, &object_type);
            let element_type = match self.type_of_value(&tuple) {
                Type::Tuple(elements) => elements.get(index as usize).cloned().unwrap_or(Type::ERROR),
                _ => Type::ERROR,
            };
            self.push_instruction(Instruction::IndexTuple { tuple, index }, element_type)
        }
    }

    /// Retain (bump the refcount of) `value` when `expr` is a shared or heap-env-closure *place* --
    /// a variable or field access naming an existing handle. Copying a handle into a new owning
    /// location (a call argument under callee-owns, a binding, a `:=` store, a stored field) takes
    /// a reference. A `Call`/`Constructor`/`Lambda` result is freshly allocated (already count 1),
    /// i.e. a move, so it is not retained.
    fn retain_if_shared_place(&mut self, expr: ExprId, value: Value) {
        let is_place = matches!(&self.context()[expr], cst::Expr::Variable(_) | cst::Expr::MemberAccess(_));
        if is_place {
            self.emit_place_retain(expr, value);
        }
    }

    /// Emit the appropriate refcount retain for a place `expr` producing `value`: a shared handle is
    /// the pointer itself (`RcRetain`); a heap-env closure carries its refcount on its environment
    /// pointer (`IndexTuple(closure, 1)` + null-safe `RetainClosureEnv`). Everything else is a no-op.
    /// The caller has already established `expr` is a place worth retaining.
    fn emit_place_retain(&mut self, expr: ExprId, value: Value) {
        if self.expr_tc_is_shared(expr) {
            self.push_instruction(Instruction::RcRetain(value), Type::UNIT);
        } else if self.expr_tc_is_heap_env_closure(expr) {
            let env = self.push_instruction(Instruction::IndexTuple { tuple: value, index: 1 }, Type::POINTER);
            self.push_instruction(Instruction::RetainClosureEnv(env), Type::UNIT);
        }
    }

    fn call(&mut self, call: &cst::Call, id: ExprId) -> Value {
        // Intrinsics in the stdlib are written as a call `intrinsic "Name" arg1 ... argN`
        // We check for this case here since the arguments are required to lower to concrete
        // instructions rather than a function wrapper (which would be needed if this was handled
        // in the recursive `cst::Variable` case when lowering the function expression)
        if let Some(result) = self.try_lower_intrinsic(call, id) {
            return result;
        }

        let diverges = self.callee_diverges(call.function);
        let result_type = if diverges { Type::UNIT } else { self.expr_type(id) };

        // Ability method calls (both impl-style and effect-style) are routed through the
        // implicit capability tuple: the implicit cap is the last argument, and the operation
        // we want is at `op_index` within that tuple. Emit `IndexTuple cap op_index +
        // CallClosure` directly so we don't depend on the ability-method's own [DefinitionId]
        // having a globally-consistent function type (a single [DefinitionId] can be used at
        // multiple sites that unify its env differently).
        //
        // For effects, the `handle` expression replaces the cap value with a tuple of
        // operation wrappers, so the IndexTuple at the call site naturally picks up the
        // wrapper. For traits, the cap is a struct-typed impl value with the operation
        // closures as fields; the IndexTuple picks up the impl's method.
        if let cst::Expr::Variable(path_id) = &self.context()[call.function]
            && let Some((effect_op, op_index)) = self.try_resolve_ability_method(*path_id)
        {
            // A capability built by applying a constrained impl (`drop_vec drop_i32`) heap-
            // allocates each method closure's environment inside the impl function, freshly
            // per call site execution. A `Drop` capability cannot outlive its method call
            // (unlike e.g. Stream combinators, which legitimately capture their caps in
            // returned closures), so for Drop calls we lower application-shaped capability
            // arguments through `lower_drop_capability` -- collecting every nesting level's
            // environments -- and free them once the drop completes. Static impls and
            // capability parameters lower as plain variables and are never freed.
            let is_drop_call = matches!(
                self.context().path_origin(*path_id),
                Some(Origin::TopLevelDefinition(name)) if self.is_prelude_drop_ability(name.top_level_item)
            );

            let mut cap_env_frees = Vec::new();
            let arguments = mapvec(&call.arguments, |argument| {
                if is_drop_call && argument.is_implicit {
                    self.lower_drop_capability(argument.expr, &mut cap_env_frees)
                } else {
                    let value = self.expression(argument.expr);
                    // Callee-owns retain, same as the plain-call path below: an effect-op or
                    // ability-method argument lands in an owned parameter (the handler arm or
                    // impl method releases it), so a shared handle passed here needs its
                    // refcount bumped or the arm's release double-frees the caller's handle.
                    // Implicit capability tuples are excluded: they are not shared handles,
                    // and their ownership is audited separately.
                    if !argument.is_implicit {
                        self.retain_if_shared_place(argument.expr, value);
                    }
                    value
                }
            });

            let result = self.emit_ability_method_call(effect_op, op_index, arguments, result_type, diverges);
            if !diverges {
                for environment in cap_env_frees {
                    self.emit_free(environment);
                }
            }
            return result;
        }

        let function = self.expression(call.function);
        // A synthesized `release_T` call *consumes* a reference (it decrements), so its argument
        // must not be retained -- retaining would cancel the decrement and leak. Every other call is
        // callee-owns: a shared handle passed by value is a new owning location, so retain it (the
        // callee's parameter release, or the constructed value's later release, balances it).
        let is_release_call = matches!(&self.context()[call.function], cst::Expr::Variable(path)
            if matches!(self.context().path_origin(*path), Some(Origin::TopLevelDefinition(name)) if name.local_name_id == NameId::RELEASE_FUNCTION));
        let arguments = mapvec(&call.arguments, |arg| {
            let value = self.expression(arg.expr);
            if !is_release_call {
                self.retain_if_shared_place(arg.expr, value);
            }
            value
        });

        let instruction = if self.type_of_value(&function).is_closure() {
            Instruction::CallClosure { closure: function, arguments }
        } else {
            Instruction::Call { function, arguments }
        };

        let value = self.push_instruction(instruction, result_type);
        if diverges {
            self.terminate_block(TerminatorInstruction::Unreachable);
        }
        value
    }

    /// Emit the `IndexTuple cap op_index + CallClosure` sequence for an ability-method call.
    /// `arguments` must contain the operation args followed by the implicit capability value
    /// (the cap is the last argument, appended by implicit-arg resolution).
    fn emit_ability_method_call(
        &mut self, effect_op: DefinitionId, op_index: u32, mut arguments: Vec<Value>, result_type: Type, diverges: bool,
    ) -> Value {
        // Record this op→index pair so any `Handle` wrapper machinery downstream can look up
        // the slot of the op in the cap tuple.
        self.effect_op_indices.insert(effect_op, op_index);

        let cap_value = arguments.pop().expect("ability method call: no implicit cap argument");
        let cap_type = self.type_of_value(&cap_value);
        let method_type = match &cap_type {
            Type::Tuple(fields) => fields.get(op_index as usize).cloned().unwrap_or_else(|| {
                panic!("ability method call: cap tuple has no slot {op_index} (cap_type = {cap_type})")
            }),
            // Single-method abilities can collapse to a bare function when type inference
            // strips the surrounding tuple. Fall back to the call's expected result-type wiring.
            _ => cap_type.clone(),
        };

        let method =
            self.push_instruction(Instruction::IndexTuple { tuple: cap_value, index: op_index }, method_type.clone());

        let instruction = if method_type.is_closure() {
            Instruction::CallClosure { closure: method, arguments }
        } else {
            Instruction::Call { function: method, arguments }
        };

        let value = self.push_instruction(instruction, result_type);
        if diverges {
            self.terminate_block(TerminatorInstruction::Unreachable);
        }
        value
    }

    /// True if `item` is the Prelude's `Drop` ability definition.
    fn is_prelude_drop_ability(&mut self, item: TopLevelId) -> bool {
        if item.source_file != crate::name_resolution::namespace::SourceFileId::prelude() {
            return false;
        }
        let (raw_item, context) = GetItemRaw(item).get(self.compiler);
        match &raw_item.kind {
            cst::TopLevelItemKind::AbilityDefinition(ability) => context.names[ability.name].as_ref() == "Drop",
            _ => false,
        }
    }

    /// Lower a Drop call's capability argument. Application-shaped arguments (a constrained
    /// impl applied to its own capabilities, e.g. `drop_vec drop_i32`) are lowered manually,
    /// recursing into their arguments, so that every nesting level's freshly-constructed
    /// capability value is at hand: each one's method-closure environments are recorded in
    /// `env_frees` for release after the drop runs. Anything else (a static impl reference,
    /// a `{Drop t}` parameter) lowers normally and owns nothing to free.
    fn lower_drop_capability(&mut self, expr: ExprId, env_frees: &mut Vec<Value>) -> Value {
        let call = match &self.context()[expr] {
            cst::Expr::Call(call) => call.clone(),
            _ => return self.expression(expr),
        };

        let function = self.expression(call.function);
        let arguments = mapvec(&call.arguments, |argument| self.lower_drop_capability(argument.expr, env_frees));
        let result_type = self.expr_type(expr);

        let instruction = if self.type_of_value(&function).is_closure() {
            Instruction::CallClosure { closure: function, arguments }
        } else {
            Instruction::Call { function, arguments }
        };
        let capability = self.push_instruction(instruction, result_type);

        self.collect_capability_environments(capability, env_frees);
        capability
    }

    /// Record the environment pointer of each closure-shaped method in the capability tuple.
    /// Capture-less methods carry a null environment; `free(NULL)` is a no-op, so they need
    /// no special casing.
    fn collect_capability_environments(&mut self, capability: Value, env_frees: &mut Vec<Value>) {
        let capability_type = self.type_of_value(&capability);
        let Type::Tuple(fields) = &capability_type else { return };
        let fields = fields.clone();
        for (index, field) in fields.iter().enumerate() {
            if field.is_closure() {
                let closure = self.push_instruction(
                    Instruction::IndexTuple { tuple: capability, index: index as u32 },
                    field.clone(),
                );
                let environment =
                    self.push_instruction(Instruction::IndexTuple { tuple: closure, index: 1 }, Type::POINTER);
                env_frees.push(environment);
            }
        }
    }

    /// Free a heap pointer the drop machinery owns. These pointers come from [Instruction::
    /// AllocShared] (closure/capability-method environments), which carry a refcount header before
    /// the value; [Instruction::FreeShared] subtracts that offset and is null-safe (capture-less
    /// methods carry a null environment). Raw `Ptr` frees in the stdlib use a plain `free` and are
    /// unaffected.
    fn emit_free(&mut self, pointer: Value) {
        self.push_instruction(Instruction::FreeShared(pointer), Type::UNIT);
    }

    /// Like [Self::try_resolve_effect_op] but also returns the op's position within its
    /// ability's body, suitable for `IndexTuple` against the cap value.
    fn try_resolve_ability_method(&mut self, path: PathId) -> Option<(DefinitionId, u32)> {
        let origin = self.context().path_origin(path)?;
        let Origin::TopLevelDefinition(name) = origin else { return None };

        let is_ability = self.ability_defs.get(&name.top_level_item).copied().unwrap_or_else(|| {
            let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
            let is_ability = matches!(&item.kind, cst::TopLevelItemKind::AbilityDefinition(_));
            self.ability_defs.insert(name.top_level_item, is_ability);
            is_ability
        });
        if !is_ability {
            return None;
        }

        let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
        if let cst::TopLevelItemKind::AbilityDefinition(effect) = &item.kind
            && let Some(op_index) = effect.body.iter().position(|d| d.name == name.local_name_id)
        {
            let id = self.get_definition_id(&name);
            return Some((id, op_index as u32));
        }
        None
    }

    /// Looks up the callee via `path_types` rather than `expr_types`: the latter is overwritten
    /// with the post-unification expected type, which may have erased `Never`.
    fn callee_diverges(&self, callee_expr: ExprId) -> bool {
        // TODO: Test whether this and `Never` handling in MIR in general holds up for generic functions.
        // if a return type is a generic bound to Never by monomorphization we won't find it here.
        let cst::Expr::Variable(path_id) = &self.context()[callee_expr] else {
            return false;
        };
        let Some(typ) = self.types.result.maps.path_types.get(path_id) else {
            return false;
        };
        function_returns_never(typ, &self.types.bindings)
    }

    /// If `path` resolves to an effect-operation declaration, return its
    /// [DefinitionId]. Otherwise return None.
    ///
    /// We use a cache to avoid hitting inc-complete's locks for every
    /// call instruction but this can likely be improved further.
    fn try_resolve_effect_op(&mut self, path: PathId) -> Option<DefinitionId> {
        let origin = self.context().path_origin(path)?;
        let Origin::TopLevelDefinition(name) = origin else { return None };

        let is_ability = self.ability_defs.get(&name.top_level_item).copied().unwrap_or_else(|| {
            let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
            let is_ability = matches!(&item.kind, cst::TopLevelItemKind::AbilityDefinition(_));
            self.ability_defs.insert(name.top_level_item, is_ability);
            is_ability
        });
        if !is_ability {
            return None;
        }

        // Cold path: this TopLevelId is an effect definition. Verify that the
        // referenced NameId is one of its ops. Paths can also resolve to the
        // ability's type-constructor name or to a non-function field on the
        // ability (sub-ability reference)
        let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
        if let cst::TopLevelItemKind::AbilityDefinition(effect) = &item.kind
            && let Some(op_index) = effect.body.iter().position(|d| d.name == name.local_name_id)
        {
            let id = self.get_definition_id(&name);
            self.effect_op_indices.insert(id, op_index as u32);
            return Some(id);
        }
        None
    }

    fn try_find_name(&self, pattern: PatternId) -> Option<(Name, NameId)> {
        match &self.context()[pattern] {
            cst::Pattern::Error => None,
            cst::Pattern::Literal(_) => None,
            cst::Pattern::Constructor(..) => None,
            cst::Pattern::TypeAnnotation(pattern, _) => self.try_find_name(*pattern),
            cst::Pattern::Variable(name)
            | cst::Pattern::MethodName { item_name: name, .. }
            | cst::Pattern::Alias(name, _) => Some((self.context()[*name].clone(), *name)),
            cst::Pattern::Or(alts) => alts.first().and_then(|alt| self.try_find_name(*alt)),
        }
    }

    fn collect_pattern_names(&self, pattern: PatternId, out: &mut Vec<NameId>) {
        match &self.context()[pattern] {
            cst::Pattern::Error | cst::Pattern::Literal(_) => (),
            cst::Pattern::Variable(name) | cst::Pattern::MethodName { item_name: name, .. } => {
                out.push(*name);
            },
            cst::Pattern::TypeAnnotation(pattern, _) => self.collect_pattern_names(*pattern, out),
            cst::Pattern::Constructor(_, arguments) => {
                for argument in arguments.clone() {
                    self.collect_pattern_names(argument, out);
                }
            },
            // Each alternative of a valid OR-pattern binds the same names, so we only
            // need to inspect the first alternative.
            cst::Pattern::Or(alts) => {
                if let Some(alt) = alts.first() {
                    self.collect_pattern_names(*alt, out);
                }
            },
            cst::Pattern::Alias(name, inner) => {
                out.push(*name);
                self.collect_pattern_names(*inner, out);
            },
        }
    }

    /// Save the current function state, create a new function,
    /// run `f` to fill in the function's body, then restore the previous state
    /// and return the new function value.
    fn new_definition(
        &mut self, name: Name, name_id: Option<NameId>, generic_count: u32, typ: Type, f: impl FnOnce(&mut Self),
    ) -> DefinitionId {
        let state = self.start_global(name, name_id, generic_count, typ);
        f(self);
        self.end_global(state)
    }

    fn start_global(
        &mut self, name: Name, name_id: Option<NameId>, generic_count: u32, typ: Type,
    ) -> (Option<Definition>, BlockId) {
        // Safety: This function must always be paired with [Self::end_global]
        let previous_function = self.current_function.take();
        let previous_block = std::mem::replace(&mut self.current_block, BlockId::ENTRY_BLOCK);

        let definition_id = if let Some(name_id) = name_id {
            let id = self.get_definition_id(&TopLevelName::new(self.top_level_id, name_id));
            self.external.remove(&id);
            id
        } else {
            next_definition_id()
        };

        self.current_function = Some(Definition::new(name, definition_id, generic_count, typ));
        (previous_function, previous_block)
    }

    fn end_global(&mut self, start_global_state: (Option<Definition>, BlockId)) -> DefinitionId {
        // Safety: This function must always be paired with [Self::start_global]
        let finished_function = std::mem::replace(&mut self.current_function, start_global_state.0).unwrap();

        let definition_id = finished_function.id;
        self.current_block = start_global_state.1;

        self.finished_functions.insert(definition_id, finished_function);
        definition_id
    }

    fn lambda(
        &mut self, lambda: &cst::Lambda, name_id: Option<NameId>, name: Option<Name>, expr: ExprId, is_global: bool,
    ) -> Value {
        self.lambda_impl(lambda, name_id, name, expr, is_global, None)
    }

    /// When `handle_body_handler_name` is `Some(h)`, this lambda is the body of a `handle` expression.
    /// Inject a prelude that loads the capability from the coroutine's user_data and binds it
    /// to `h` in `local_variables` so the body's references to `h` resolve.
    fn lambda_impl(
        &mut self, lambda: &cst::Lambda, name_id: Option<NameId>, name: Option<Name>, expr: ExprId, is_global: bool,
        handle_body_handler_name: Option<NameId>,
    ) -> Value {
        let name = name.unwrap_or_else(|| Arc::new("lambda".to_string()));
        let full_type = self.convert_expr_type(expr);
        let Type::Function(function_type) = &full_type else { unreachable!("Lambda does not have a function type") };

        let is_move = self.context().is_move_closure(expr);

        let mutable_captures: rustc_hash::FxHashSet<NameId> =
            if let Some(free_vars) = self.context().get_closure_environment(expr) {
                free_vars.iter().filter(|v| self.mutable_locals.contains(v)).copied().collect()
            } else {
                Default::default()
            };

        // Lambdas aren't really generic but they may capture generics from their containing
        // function. Marking them as generic here effectively hoists out the generic parameters.
        // It also means we have to instantiate lambdas when they're used with the current generic
        // parameters.
        let generics_count = self.generics_in_scope.len() as u32;
        let old_scope = std::mem::take(&mut self.local_variables);
        let old_mutables = std::mem::take(&mut self.mutable_locals);
        let old_loop_targets = std::mem::take(&mut self.loop_targets);

        let id = self.new_definition(name.clone(), name_id, generics_count, full_type.clone(), |this| {
            for (i, parameter) in lambda.parameters.iter().enumerate() {
                let parameter_type = &this.types.result.maps.pattern_types[&parameter.pattern];
                this.push_parameter(this.convert_type(parameter_type, None));

                let parameter_value = Value::Parameter(this.current_block, i as u32);
                this.bind_pattern(parameter.pattern, parameter_value);

                if parameter.is_mutable
                    && let Some((_, name_id)) = this.try_find_name(parameter.pattern)
                {
                    let alloc = this.push_instruction(Instruction::StackAlloc(parameter_value), Type::POINTER);
                    this.local_variables.insert(name_id, alloc);
                    this.mutable_locals.insert(name_id);
                }
            }

            let env_is_pointer =
                matches!(function_type.environment, Type::Primitive(crate::mir::PrimitiveType::Pointer));
            let free_vars = this.context().get_closure_environment(expr);
            let needs_env_param = free_vars.is_some() || env_is_pointer;
            if needs_env_param {
                if let Some(env) = function_type.environment() {
                    this.push_parameter(env.clone());
                }

                if let Some(free_vars) = free_vars {
                    let environment = Value::Parameter(this.current_block, lambda.parameters.len() as u32);
                    this.unpack_closure_environment(free_vars.iter().copied(), environment);

                    // For regular closures, mutable captures are pointers (by reference).
                    if !is_move {
                        for var in free_vars.iter() {
                            if mutable_captures.contains(var) {
                                this.mutable_locals.insert(*var);
                            }
                        }
                    }
                }
            }

            // For a `handle` expression's body lambda, bind `h` to a placeholder
            // [Instruction::Capability]. The lowering passes are responsible for replacing it:
            // [crate::mir::effects::effect_lowering] expands it into a coroutine `user_data`
            // fetch, while [crate::mir::effects::tail_resume_optimization] rewrites it to refer
            // to a directly-built capability tuple.
            if let Some(handler_name) = handle_body_handler_name {
                let h_tc_type = this.types.result.maps.name_types[&handler_name].clone();
                let h_type = this.convert_type(&h_tc_type, None);
                let cap = this.push_instruction(Instruction::Capability, h_type);
                this.local_variables.insert(handler_name, cap);
            }

            // Pre-populate the self-reference so recursive calls within the body
            // (e.g. `recur` in desugared loop expressions) can resolve via Origin::Local.
            // The entry is discarded automatically when `old_scope` is restored after
            // `new_definition` returns.
            if let Some(self_name_id) = name_id {
                let self_def_id = this.current_function.as_ref().unwrap().id;
                let mut self_value = Value::Definition(self_def_id);
                if needs_env_param {
                    let environment = Value::Parameter(this.current_block, lambda.parameters.len() as u32);
                    self_value = this.push_instruction(
                        Instruction::PackClosure { function: self_value, environment },
                        full_type.clone(),
                    );
                }
                this.local_variables.insert(self_name_id, self_value);
            }

            let return_value = this.expression(lambda.body);
            this.terminate_block(TerminatorInstruction::Return(return_value));
        });

        self.local_variables = old_scope;
        self.mutable_locals = old_mutables;
        self.loop_targets = old_loop_targets;

        // Generic lambdas inherit generics from the surrounding context; instantiate manually.
        let mut value = self.make_definition_value(id, name, full_type.clone());
        if !is_global && !self.generics_in_scope.is_empty() {
            let bindings = Arc::new(mapvec(0..self.generics_in_scope.len() as u32, |i| Type::Generic(Generic(i))));
            value = self.push_instruction(Instruction::Instantiate(id, bindings), full_type.clone());
        }
        let free_vars = self.context().get_closure_environment(expr).cloned();
        let env_type = function_type.environment.clone();
        let env_is_pointer = matches!(env_type, Type::Primitive(crate::mir::PrimitiveType::Pointer));
        if free_vars.is_some() || env_is_pointer {
            let environment = if let Some(free_vars) = &free_vars {
                self.pack_closure_environment(free_vars, is_move, &env_type)
            } else {
                // Pointer-env slot with no captures (e.g. an ability impl assigning a plain function):
                // use a null pointer for the env. Transmute from Unit so constant-folding works
                // even when this PackClosure ends up in a global initializer.
                self.push_instruction(Instruction::Transmute(Value::Unit), Type::POINTER)
            };
            value = self.push_instruction(Instruction::PackClosure { function: value, environment }, full_type);
        }
        value
    }

    /// Packs each given variable into a closure environment.
    /// When `env_type` is a pointer, the capture tuple is heap-allocated (via [Instruction::AllocShared])
    /// and the returned value is the resulting pointer. Otherwise returns the tuple directly.
    fn pack_closure_environment(&mut self, free_vars: &BTreeSet<NameId>, is_move: bool, env_type: &Type) -> Value {
        assert!(!free_vars.is_empty());

        let values = mapvec(free_vars, |var| {
            let value = *self.local_variables.get(var).unwrap();

            if is_move && self.mutable_locals.contains(var) {
                // `move` closures capture mutable variables by value: dereference the
                // StackAlloc pointer to load the current value into the environment.
                let tc_type = &self.types.result.maps.name_types[var];
                let val_type = self.convert_type(tc_type, None);
                self.push_instruction(Instruction::Deref(value), val_type)
            } else {
                value
            }
        });
        let types = mapvec(&values, |value| self.type_of_value(value));
        let tuple = self.push_instruction(Instruction::MakeTuple(values), Type::tuple(types));

        if matches!(env_type, Type::Primitive(crate::mir::PrimitiveType::Pointer)) {
            self.push_instruction(Instruction::AllocShared(tuple), Type::POINTER)
        } else {
            tuple
        }
    }

    /// Unpack a closure environment parameter, binding each captured name to its value.
    /// When the env value is a `Pointer` (ability-method-style heap env), it is dereferenced first
    /// to recover the capture tuple. Expects `free_vars` to be non-empty.
    fn unpack_closure_environment(
        &mut self, free_vars: impl ExactSizeIterator<Item = NameId> + Clone, environment: Value,
    ) {
        let env_value =
            if matches!(self.type_of_value(&environment), Type::Primitive(crate::mir::PrimitiveType::Pointer)) {
                let field_types: Vec<Type> = free_vars
                    .clone()
                    .map(|var| {
                        let tc_type = &self.types.result.maps.name_types[&var];
                        self.convert_type(tc_type, None)
                    })
                    .collect();
                let tuple_type = Type::tuple(field_types);
                self.push_instruction(Instruction::Deref(environment), tuple_type)
            } else {
                environment
            };

        let Type::Tuple(env_fields) = self.type_of_value(&env_value) else { unreachable!() };
        assert_eq!(env_fields.len(), free_vars.len());

        for (i, (var, env_field)) in free_vars.zip(env_fields.iter().cloned()).enumerate() {
            let index = Instruction::IndexTuple { tuple: env_value, index: i as u32 };
            let result = self.push_instruction(index, env_field);
            let existing = self.local_variables.insert(var, result);
            assert!(existing.is_none(), "Closure is overwriting values from the outer scope");
        }
    }

    fn while_(&mut self, while_: &cst::While) -> Value {
        let header = self.push_block_no_params();
        let body = self.push_block_no_params();
        let exit = self.push_block_no_params();

        self.terminate_block(TerminatorInstruction::jmp_no_args(header));

        self.switch_to_block(header);
        let cond = self.expression(while_.condition);
        self.terminate_block(TerminatorInstruction::if_(cond, body, exit, exit));

        self.switch_to_block(body);
        self.loop_targets.push((header, exit));
        let _ = self.expression(while_.body);
        self.loop_targets.pop();
        self.terminate_block(TerminatorInstruction::jmp_no_args(header));

        self.switch_to_block(exit);
        Value::Unit
    }

    fn for_(&mut self, for_: &cst::For) -> Value {
        let variable_type = self.expr_type(for_.start);
        let kind = match &variable_type {
            Type::Primitive(crate::mir::PrimitiveType::Int(k)) => *k,
            _ => unreachable!("for-loop range was not inferred to an integer type: {variable_type:?}"),
        };

        let start_value = self.expression(for_.start);
        let end_value = self.expression(for_.end);

        let header = self.push_block(vec![variable_type.clone()]);
        let body = self.push_block_no_params();
        let step = self.push_block_no_params();
        let exit = self.push_block_no_params();

        let variable = Value::Parameter(header, 0);
        self.local_variables.insert(for_.variable, variable);

        self.terminate_block(TerminatorInstruction::jmp(header, start_value));

        self.switch_to_block(header);
        let cmp = if kind.is_signed() {
            Instruction::LessSigned(variable, end_value)
        } else {
            Instruction::LessUnsigned(variable, end_value)
        };
        let cond = self.push_instruction(cmp, Type::BOOL);
        self.terminate_block(TerminatorInstruction::if_(cond, body, exit, exit));

        self.switch_to_block(body);
        self.loop_targets.push((step, exit));
        let _ = self.expression(for_.body);
        self.loop_targets.pop();
        self.terminate_block(TerminatorInstruction::jmp_no_args(step));

        self.switch_to_block(step);
        let one = Self::integer(Integer::positive(1), kind);
        let variable_plus_one = self.push_instruction(Instruction::AddInt(variable, one), variable_type);
        self.terminate_block(TerminatorInstruction::jmp(header, variable_plus_one));

        self.switch_to_block(exit);
        Value::Unit
    }

    fn break_(&mut self, expr: ExprId) -> Value {
        self.lower_pre_exit_drops(expr);
        let exit = self.loop_targets.last().expect("`break` outside of a loop").1;
        self.terminate_block(TerminatorInstruction::jmp_no_args(exit));
        Value::Error
    }

    fn continue_(&mut self, expr: ExprId) -> Value {
        self.lower_pre_exit_drops(expr);
        let cont = self.loop_targets.last().expect("`continue` outside of a loop").0;
        self.terminate_block(TerminatorInstruction::jmp_no_args(cont));
        Value::Error
    }

    /// Auto-drop: lower the synthesized drops recorded for this exit edge (break/continue).
    fn lower_pre_exit_drops(&mut self, expr: ExprId) {
        if let Some(drops) = self.context().pre_exit_drops(expr) {
            for drop_call in drops.clone() {
                self.expression(drop_call);
            }
        }
    }

    fn if_(&mut self, if_: &cst::If, expr: ExprId) -> Value {
        let condition = self.expression(if_.condition);

        // Auto-drop: an else-less `if` whose then-branch moves values needs a real else
        // block so the false edge can drop what it still owns.
        let implicit_else_drops = self.context().implicit_else_drops(expr).cloned();

        let then = self.push_block_no_params();
        let else_ = self.push_block_no_params();
        let end = if if_.else_.is_some() || implicit_else_drops.is_some() {
            self.push_block_no_params()
        } else {
            else_
        };
        self.terminate_block(TerminatorInstruction::if_(condition, then, else_, end));

        self.switch_to_block(then);
        let then_value = self.expression(if_.then);

        if let Some(else_expr) = if_.else_ {
            let result_type = self.expr_type(expr);
            self.terminate_block(TerminatorInstruction::jmp(end, then_value));

            self.switch_to_block(else_);
            let else_value = self.expression(else_expr);
            self.terminate_block(TerminatorInstruction::jmp(end, else_value));
            self.switch_to_block(end);
            self.push_parameter(result_type);
            Value::Parameter(end, 0)
        } else {
            self.terminate_block(TerminatorInstruction::jmp_no_args(end));
            if let Some(drops) = implicit_else_drops {
                self.switch_to_block(else_);
                for drop_call in drops {
                    self.expression(drop_call);
                }
                self.terminate_block(TerminatorInstruction::jmp_no_args(end));
            }
            self.switch_to_block(end);
            Value::Unit
        }
    }

    fn match_(&mut self, expr: ExprId) -> Value {
        match self.context().decision_tree(expr) {
            Some((define_match_var, tree)) => {
                self.expression(*define_match_var);
                self.decision_tree(tree.clone(), expr)
            },
            None => Value::Error,
        }
    }

    fn decision_tree(&mut self, tree: DecisionTree, match_expr: ExprId) -> Value {
        match tree {
            DecisionTree::Success(expr) => self.expression(expr),
            // Expect an error to already be issued
            DecisionTree::Failure { .. } => Value::Error,
            DecisionTree::Guard { condition, then, else_ } => self.match_if_guard(condition, then, *else_, match_expr),
            DecisionTree::Switch(tag, cases, else_) => self.switch(tag, cases, else_, match_expr),
        }
    }

    /// Almost identical to an if-then-else, the main difference being the else is required
    /// and is of type [DecisionTree]
    fn match_if_guard(
        &mut self, condition: ExprId, then_expr: ExprId, else_tree: DecisionTree, match_expr: ExprId,
    ) -> Value {
        let then = self.push_block_no_params();
        let else_ = self.push_block_no_params();

        let result_type = self.expr_type(match_expr);
        let end = self.push_block(vec![result_type]);

        let condition = self.expression(condition);
        self.terminate_block(TerminatorInstruction::if_(condition, then, else_, else_));

        self.switch_to_block(then);
        let then_value = self.expression(then_expr);
        self.terminate_block(TerminatorInstruction::jmp(end, then_value));

        self.switch_to_block(else_);
        let else_value = self.decision_tree(else_tree, match_expr);
        self.terminate_block(TerminatorInstruction::jmp(end, else_value));

        self.switch_to_block(end);
        Value::Parameter(end, 0)
    }

    fn switch(&mut self, tag: PathId, cases: Vec<Case>, else_: Option<Box<DecisionTree>>, match_expr: ExprId) -> Value {
        let path_type = self.types.result.maps.path_types[&tag].clone();
        let value_being_matched = self.variable(tag);
        let value_being_matched = self.deref_if_shared(value_being_matched, &path_type);
        let int_value = self.extract_tag_value(value_being_matched);
        let start = self.current_block;

        // The case key must be the actual value to compare against `int_value`. For variants
        // the tag value is the variant index; for int constants it's the constant itself.
        let case_blocks = mapvec(cases.iter(), |case| {
            let key = match &case.constructor {
                Constructor::False | Constructor::Unit => 0,
                Constructor::True => 1,
                Constructor::Int(value) => *value as u32,
                Constructor::Variant(_, variant_index) => *variant_index as u32,
                // Range constructors aren't lowered through a Switch directly; if we ever
                // encounter one here it's a compiler bug.
                Constructor::Range(_, _) => unreachable!("Range constructor in MIR switch lowering"),
            };
            (key, (self.push_block_no_params(), None))
        });

        let result_type = self.expr_type(match_expr);
        let end = self.push_block(vec![result_type]);

        for ((_, (case_block, _)), case) in case_blocks.iter().zip(cases) {
            self.switch_to_block(*case_block);

            if !case.arguments.is_empty() {
                let Constructor::Variant(_, variant_index) = &case.constructor else {
                    unreachable!("For this constructor to define arguments it must be a Constructor::Variant")
                };

                // Cast the whole value being matched `(tag, union)` to `(tag, this_variant)`
                // and extract the variant from the tuple.
                let variant = self.extract_variant(value_being_matched, *variant_index);
                let variant_type = self.type_of_value(&variant);

                // And for each variable, extract the relevant field of the variant
                for (i, argument) in case.arguments.iter().enumerate() {
                    if let Some(origin) = self.context().path_origin(*argument) {
                        let field_type = Self::tuple_field_type(&variant_type, i);
                        let index_tuple = Instruction::IndexTuple { tuple: variant, index: i as u32 };
                        let field = self.push_instruction(index_tuple, field_type);
                        self.define_variable(origin, field);
                    }
                }
            }

            let result = self.decision_tree(case.body, match_expr);
            self.terminate_block(TerminatorInstruction::jmp(end, result));
        }

        let else_block = self.push_block_no_params();
        self.switch_to_block(else_block);
        let terminator = match else_ {
            Some(else_) => {
                let result = self.decision_tree(*else_, match_expr);
                TerminatorInstruction::jmp(end, result)
            },
            None => TerminatorInstruction::Unreachable,
        };
        self.terminate_block(terminator);

        self.switch_to_block(start);
        self.terminate_block(TerminatorInstruction::Switch {
            int_value,
            cases: case_blocks,
            else_: (else_block, None),
            end,
        });
        self.switch_to_block(end);
        Value::Parameter(end, 0)
    }

    fn extract_tag_value(&mut self, value_being_matched: Value) -> Value {
        match self.type_of_value(&value_being_matched) {
            Type::Primitive(_) => value_being_matched,
            Type::Tuple(fields) => {
                if fields.is_empty() {
                    unreachable!("Cannot match on an empty tuple")
                }
                // Tagged unions have the form `(u8_tag, Union[...])`. For product
                // types (pairs, structs) there is always exactly one case, so return
                // a constant 0 so the switch always selects case 0.
                if fields.len() == 2 && matches!(fields[1], Type::Union(_)) {
                    let tag_type = fields[0].clone();
                    let instruction = Instruction::IndexTuple { tuple: value_being_matched, index: 0 };
                    self.push_instruction(instruction, tag_type)
                } else {
                    Value::tag_value(0)
                }
            },
            Type::Union(_) => unreachable!("Cannot match on a raw union type"),
            Type::Function(_) => unreachable!("Cannot match on a function type"),
            Type::Generic(_) => unreachable!("Cannot match on a generic type"),
            Type::Array { .. } => unreachable!("Cannot match on an array type"),
            Type::U32(_) => unreachable!("Cannot match on a type-level integer"),
        }
    }

    /// Cast & Extract the variant value from the given `(tag, union)` tuple,
    /// or return the product type value directly if it has no union tag.
    /// Returns the variant (as a tuple, no longer a union).
    fn extract_variant(&mut self, value_being_matched: Value, variant_index: usize) -> Value {
        let fields = match self.type_of_value(&value_being_matched) {
            Type::Tuple(fields) => fields,
            _ => unreachable!("Only `(tag, union)` tuples may have fields to extract"),
        };

        // Tagged unions have the form `(u8_tag, Union[...])` while
        // product types are plain tuples and should be returned directly
        if fields.len() != 2 || !matches!(fields[1], Type::Union(_)) {
            return value_being_matched;
        }

        let union_type = fields[1].clone();
        let Type::Union(variants) = &union_type else { unreachable!() };

        let variant_type = variants
            .get(variant_index)
            .unwrap_or_else(|| {
                unreachable!("Expected variant index {variant_index} but only had {} variants", variants.len())
            })
            .clone();

        let extract_union = Instruction::IndexTuple { tuple: value_being_matched, index: 1 };
        let union = self.push_instruction(extract_union, union_type);
        self.push_instruction(Instruction::Transmute(union), variant_type)
    }

    fn handle(&mut self, handle: &cst::Handle, expr: ExprId) -> Value {
        let result_type = self.expr_type(expr);

        // The parser wraps the handled expression in `fn () -> <body>` so it can serve as a
        // coroutine entry. Build that body lambda with a prelude that loads `h` from the
        // coroutine's user_data.
        let body = match &self.context()[handle.expression] {
            cst::Expr::Lambda(body_lambda) => {
                let body_lambda = body_lambda.clone();
                self.lambda_impl(&body_lambda, None, None, handle.expression, false, Some(handle.handler_name))
            },
            _ => unreachable!("handle.expression should be a Lambda after parsing"),
        };

        let cases = mapvec(&handle.cases, |(pattern, branch)| {
            let effect_op =
                self.try_resolve_effect_op(pattern.function).expect("Couldn't find effect op in MIR handle");
            let handler = self.expression(*branch);
            crate::mir::HandlerCase { effect_op, handler }
        });

        self.push_instruction(Instruction::Handle { body, cases }, result_type)
    }

    fn reference(&mut self, reference: &cst::Reference) -> Value {
        let rhs = reference.rhs;
        let context = self.context();

        // If the RHS is a locally mutable variable, its value in local_variables
        // is already the StackAlloc pointer.
        if let cst::Expr::Variable(path_id) = &context[rhs] {
            let path_id = *path_id;
            if let Some(Origin::Local(name)) = context.path_origin(path_id)
                && self.mutable_locals.contains(&name)
            {
                return *self.local_variables.get(&name).expect("mutable local variable not found in local_variables");
            }
        }

        // Reborrow: if the rhs already has a reference type its value is already a pointer
        // (e.g. a `mut Array` parameter), so `mut x` reborrows that pointer directly instead of
        // taking the address of the local holding it. Mirrors the type-level reborrow in
        // `check_reference`.
        let rhs_type = self.types.result.maps.expr_types[&rhs].follow(&self.types.bindings);
        if rhs_type.reference_element(&self.types.bindings).is_some() {
            return self.expression(rhs);
        }

        if let cst::Expr::MemberAccess(_) = &self.context()[rhs]
            && self.reference_target_is_addressable(rhs)
        {
            return self.lhs_as_pointer(rhs);
        }

        // For all other cases (non-mutable local, temporary): evaluate the expression and
        // allocate a new stack slot for it.
        let value = self.expression(rhs);
        self.push_instruction(Instruction::StackAlloc(value), Type::POINTER)
    }

    /// Returns true if `expr` is a field-access chain (or annotated variant) rooted at a
    /// mutable local.
    fn reference_target_is_addressable(&self, expr: ExprId) -> bool {
        match &self.context()[expr] {
            cst::Expr::Variable(path_id) => {
                let path_id = *path_id;
                if matches!(self.context().path_origin(path_id), Some(Origin::Local(name)) if self.mutable_locals.contains(&name)) {
                    return true;
                }
                // A `shared mut` root: field addresses reached through it are genuine (they go
                // via the shared pointer), even though the handle itself is an immutable `=`
                // binding rather than a `mutable_local`. Mirrors the shared-mut short-circuit in
                // `lhs_as_pointer`. Without this, `mut shared_handle.field` fell to the StackAlloc
                // copy path and the callee mutated a throwaway.
                let root_type = self.types.result.maps.expr_types[&expr].follow(&self.types.bindings);
                self.shared_mut_inner_layout_of(root_type).is_some()
            },
            cst::Expr::MemberAccess(ma) => self.reference_target_is_addressable(ma.object),
            cst::Expr::TypeAnnotation(ta) => self.reference_target_is_addressable(ta.lhs),
            _ => false,
        }
    }

    fn lhs_as_pointer(&mut self, lhs: ExprId) -> Value {
        // Storing a shared mut type should mutate the inner element
        let lhs_type = self.types.result.maps.expr_types[&lhs].follow(&self.types.bindings);
        let shared_mut = self.shared_mut_inner_layout_of(lhs_type).is_some();
        if shared_mut {
            return self.expression(lhs);
        }

        let context = self.context();
        let lhs_kind = match &context[lhs] {
            cst::Expr::Variable(path_id) => {
                let path_id = *path_id;
                match context.path_origin(path_id) {
                    Some(Origin::Local(name)) => LhsKind::LocalVar(name),
                    _ => LhsKind::Other,
                }
            },
            cst::Expr::Call(call) => LhsKind::DerefCall(call.arguments[0].expr),
            cst::Expr::TypeAnnotation(ta) => LhsKind::Annotation(ta.lhs),
            cst::Expr::MemberAccess(ma) => LhsKind::FieldAccess(ma.object, lhs),
            _ => LhsKind::Other,
        };
        match lhs_kind {
            LhsKind::LocalVar(name) => {
                *self.local_variables.get(&name).expect("lhs_as_pointer: mutable local variable not found")
            },
            LhsKind::DerefCall(ptr_expr) => self.expression(ptr_expr),
            LhsKind::Annotation(inner) => self.lhs_as_pointer(inner),
            LhsKind::FieldAccess(object_expr, field_expr) => {
                let struct_ptr = self.lhs_as_pointer(object_expr);
                let index = self.context().member_access_index(field_expr).unwrap_or(u32::MAX);

                // If the object has a reference type, use the inner type for GEP.
                let struct_type = self.types.result.maps.expr_types[&object_expr].follow(&self.types.bindings);
                let struct_type = if let Some(inner) = self.shared_mut_inner_layout_of(struct_type) {
                    inner
                } else if let Some(element) = struct_type.reference_or_pointer_element(&self.types.bindings) {
                    self.convert_type(element, None)
                } else {
                    self.convert_type(struct_type, None)
                };
                self.push_instruction(Instruction::GetFieldPtr { struct_ptr, struct_type, index }, Type::POINTER)
            },
            LhsKind::Other => todo!("unhandled assignment LHS"),
        }
    }

    fn assignment(&mut self, assignment: &cst::Assignment, expr: ExprId) -> Value {
        let pointer = self.lhs_as_pointer(assignment.lhs);

        let value = if let Some((_, op_expr)) = assignment.op {
            // Compound assignment: load current value, apply operator, then store.
            // The LHS is evaluated only once via lhs_as_pointer above.
            let value_type = self.compound_assign_value_type(assignment.lhs);
            let current = self.push_instruction(Instruction::Deref(pointer), value_type.clone());
            let rhs = self.expression(assignment.rhs);

            let function = self.expression(op_expr);
            let instruction = if self.type_of_value(&function).is_closure() {
                Instruction::CallClosure { closure: function, arguments: vec![current, rhs] }
            } else {
                Instruction::Call { function, arguments: vec![current, rhs] }
            };
            self.push_instruction(instruction, value_type)
        } else {
            let rhs = self.expression(assignment.rhs);

            // Overwriting a whole `shared mut` cell: copy the RHS's contents into the LHS rather than rebinding the pointer.
            let lhs_type = self.types.result.maps.expr_types[&assignment.lhs].follow(&self.types.bindings);
            match self.shared_mut_inner_layout_of(lhs_type) {
                Some(inner_layout) => self.push_instruction(Instruction::Deref(rhs), inner_layout),
                None => rhs,
            }
        };

        // Auto-drop: the overwrite drop of the old value runs after the RHS is evaluated
        // and before the store (Rust ordering; self-assignment never records one).
        if let Some(drops) = self.context().pre_exit_drops(expr) {
            for drop_call in drops.clone() {
                self.expression(drop_call);
            }
        }

        self.push_instruction(Instruction::Store { pointer, value }, Type::UNIT);
        Value::Unit
    }

    /// Get the value type for the LHS of a compound assignment.
    /// If the LHS has a reference type, returns the inner element type.
    fn compound_assign_value_type(&self, lhs: ExprId) -> Type {
        let lhs_type = &self.types.result.maps.expr_types[&lhs];

        match lhs_type.reference_element(&self.types.bindings) {
            Some((_, element)) => self.convert_type(&element, None),
            None => self.convert_type(lhs_type, None),
        }
    }

    fn constructor(&mut self, constructor: &cst::Constructor, expr: ExprId) -> Value {
        // Side-effects are executed in source order but the type must
        // be packed in declaration order. So re-order fields afterward.
        let mut fields = mapvec(&constructor.fields, |(name, field)| (*name, self.expression(*field)));

        // We must be careful here so that we can still produce MIR even if type-checking failed
        let no_order = BTreeMap::new();
        let field_order = self.context().constructor_field_order(expr).unwrap_or(&no_order);
        fields.sort_unstable_by_key(|(name, _)| field_order.get(name).unwrap_or(&0));

        // For ability impls, the struct's MIR type tells us each field's expected closure shape.
        // If a field receives a bare function (env = NoClosureEnv) where a `Ptr Unit`-env closure
        // is required, pack it with a null pointer so the produced value matches.
        let struct_type = self.convert_expr_type(expr);
        let expected_field_types: Vec<Type> =
            if let Type::Tuple(fields) = &struct_type { fields.iter().cloned().collect() } else { Vec::new() };

        let field_values = mapvec(fields.iter().enumerate(), |(i, (_n, v))| {
            self.coerce_field_to_pointer_env(*v, expected_field_types.get(i))
        });
        let tuple_type = Type::Tuple(Arc::new(mapvec(&field_values, |v| self.type_of_value(v))));

        self.push_instruction(Instruction::MakeTuple(field_values), tuple_type)
    }

    fn coerce_field_to_pointer_env(&mut self, value: Value, expected: Option<&Type>) -> Value {
        let Some(expected) = expected else { return value };
        let Type::Function(expected_fn) = expected else { return value };
        if !matches!(expected_fn.environment, Type::Primitive(crate::mir::PrimitiveType::Pointer)) {
            return value;
        }
        let actual = self.type_of_value(&value);
        let Type::Function(actual_fn) = actual else { return value };
        if !matches!(actual_fn.environment, Type::Primitive(crate::mir::PrimitiveType::NoClosureEnv)) {
            return value;
        }

        // Resolve `value` down to a (DefinitionId, optional GenericBindings) pair we can
        // re-reference inside a freshly-generated wrapper definition. If the source value
        // isn't a definition reference (e.g. it's a synthesized instruction with no
        // top-level identity) we can't safely build a wrapper here, so fall back to packing
        // the raw fn-ptr; downstream may still hit a shape mismatch but most cases (impls
        // like `cast = transmute` / `print = print_float`) resolve to Definition values.
        let (inner_def_id, inner_bindings) = match value {
            Value::Definition(id) => (id, None),
            Value::InstructionResult(iid) => {
                let bindings = match &self.current_function.as_ref().unwrap().instructions[iid] {
                    Instruction::Instantiate(id, bindings) => Some((*id, Some(bindings.clone()))),
                    Instruction::Id(Value::Definition(id)) => Some((*id, None)),
                    _ => None,
                };
                match bindings {
                    Some(p) => p,
                    None => {
                        let null_ptr = self.push_instruction(Instruction::Transmute(Value::Unit), Type::POINTER);
                        return self.push_instruction(
                            Instruction::PackClosure { function: value, environment: null_ptr },
                            expected.clone(),
                        );
                    },
                }
            },
            _ => return value,
        };

        let generic_count = self.generics_in_scope.len() as u32;
        let wrapper_type = expected.clone();
        let expected_params = expected_fn.parameters.clone();
        let expected_env = expected_fn.environment.clone();
        let expected_return = expected_fn.return_type.clone();
        let inner_typ = actual_fn.as_ref().clone();
        let inner_typ = Type::Function(Arc::new(inner_typ));
        let wrapper_id = self.new_definition(
            Arc::new("ability_field_wrapper".to_string()),
            None,
            generic_count,
            wrapper_type.clone(),
            |this| {
                for pt in &expected_params {
                    this.push_parameter(pt.clone());
                }
                this.push_parameter(expected_env.clone());
                let forward_args =
                    mapvec(0..expected_params.len(), |j| Value::Parameter(BlockId::ENTRY_BLOCK, j as u32));
                let inner_value = if let Some(bindings) = inner_bindings {
                    this.push_instruction(Instruction::Instantiate(inner_def_id, bindings), inner_typ.clone())
                } else {
                    Value::Definition(inner_def_id)
                };
                let result = this.push_instruction(
                    Instruction::Call { function: inner_value, arguments: forward_args },
                    expected_return.clone(),
                );
                this.terminate_block(TerminatorInstruction::Return(result));
            },
        );

        let wrapper_value =
            self.make_definition_value(wrapper_id, Arc::new("ability_field_wrapper".to_string()), wrapper_type.clone());
        let null_ptr = self.push_instruction(Instruction::Transmute(Value::Unit), Type::POINTER);
        self.push_instruction(Instruction::PackClosure { function: wrapper_value, environment: null_ptr }, wrapper_type)
    }

    fn quoted(&self, _quoted: &cst::Quoted) -> Value {
        unreachable!("Should never convert a Quoted expr to mir")
    }

    fn return_(&mut self, returned_expression: ExprId) -> Value {
        let value = self.expression(returned_expression);
        // Auto-drop: scope-exit drops for this return edge run after the returned value is
        // computed and before the Return terminator.
        if let Some(drops) = self.context().pre_exit_drops(returned_expression) {
            for drop_call in drops.clone() {
                self.expression(drop_call);
            }
        }
        self.terminate_block(TerminatorInstruction::Return(value));
        // TODO: We'll need to try to filter these return blocks from
        // matches & ifs, and potentially check for instructions after returns.
        Value::Error
    }

    fn extern_(&mut self, extern_: &cst::Extern, id: ExprId) -> Value {
        let typ = self.convert_expr_type(id);
        self.push_instruction(Instruction::Extern(extern_.name.clone()), typ)
    }

    /// Bind the given value to the given pattern
    fn bind_pattern(&mut self, pattern: PatternId, value: Value) {
        match &self.context()[pattern] {
            cst::Pattern::Error => unreachable!("Error pattern encountered in bind_pattern"),
            cst::Pattern::Variable(name) => {
                // This may be `None` if we had errors during name resolution
                if let Some(origin) = self.context().name_origin(*name) {
                    self.define_variable(origin, value);
                }
            },
            cst::Pattern::Literal(_) => (),
            cst::Pattern::Constructor(_type, arguments) => {
                let pattern_tc_type = self.types.result.maps.pattern_types[&pattern].clone();
                let value = self.deref_if_shared(value, &pattern_tc_type);
                match self.type_of_value(&value) {
                    Type::Union(_variants) => todo!("Deconstruct union"),
                    Type::Tuple(fields) => {
                        for (i, (field_type, argument)) in fields.iter().zip(arguments).enumerate() {
                            let instruction = Instruction::IndexTuple { tuple: value, index: i as u32 };
                            let field = self.push_instruction(instruction, field_type.clone());
                            self.bind_pattern(*argument, field);
                        }
                    },
                    other => unreachable!("Expected tuple or union when deconstructing pattern, found {other}"),
                }
            },
            cst::Pattern::TypeAnnotation(pattern, _) => self.bind_pattern(*pattern, value),
            cst::Pattern::MethodName { type_name: _, item_name } => {
                if let Some(origin) = self.context().name_origin(*item_name) {
                    self.define_variable(origin, value);
                }
            },
            cst::Pattern::Or(_) => {
                unreachable!("OR-patterns must be expanded by the decision tree compiler before MIR lowering")
            },
            cst::Pattern::Alias(name, inner) => {
                if let Some(origin) = self.context().name_origin(*name) {
                    self.define_variable(origin, value.clone());
                }
                self.bind_pattern(*inner, value);
            },
        }
    }

    fn finish(self) -> Mir {
        Mir {
            definitions: self.finished_functions,
            externals: self.external,
            preserved_op_indices: self.effect_op_indices,
        }
        .remove_internal_externs()
    }

    /// Sets [self.generics_in_scope] to a map mapping each generic from the given type to a
    /// `[mir::Generic]` used when translating [type_inference::types::Type]s into [crate::mir::Type]s.
    fn set_generics_in_scope(&mut self, definition_type: &TCType) {
        use type_inference::types::Type;
        self.generics_in_scope.clear();

        if let Type::Forall(generics, _) = definition_type {
            for (i, generic) in generics.iter().enumerate() {
                self.generics_in_scope.insert(*generic, Generic(i as u32));
            }
        }
    }

    /// For type definitions we need to define their constructors
    fn type_definition(&mut self, type_definition: &cst::TypeDefinition) {
        let constructors = match &type_definition.body {
            cst::TypeDefinitionBody::Struct(_) => vec![(type_definition.name, None)],
            cst::TypeDefinitionBody::Enum(variants) => {
                if variants.len() == 1 {
                    // `type_body` translates single constructor enums into products, we need to mirror that here
                    vec![(variants[0].0, None)]
                } else {
                    mapvec(variants.iter().enumerate(), |(i, (name, _))| (*name, Some(i.try_into().unwrap())))
                }
            },
            cst::TypeDefinitionBody::Alias(_) | cst::TypeDefinitionBody::Error => return,
        };

        for (constructor_name, tag) in constructors {
            let constructor_type = self.types.get_generalized(constructor_name);
            self.set_generics_in_scope(constructor_type);

            let parameters = match constructor_type.ignore_forall() {
                TCType::Function(function) => function.parameters.as_slice(),
                _ => &[],
            };

            let shared = type_definition.shared;
            self.define_type_constructor(constructor_name, constructor_type, parameters, tag, shared);
        }

        // A shared type owns a synthesized `release_T`: decrement the refcount, and on the last
        // reference run the pointee-drop glue then free the block.
        if type_definition.shared {
            self.define_release_function(type_definition);
        }

        // Abilities are sugar for a struct of function-typed fields, however each "field" is treated
        // as a function by the frontend so we must generate actual functions for each field such
        // that `Cast.cast` is an actual function accepting a `Cast` instance and forwarding the
        // appropriate arguments to the `cast` field.
        if type_definition.is_ability {
            self.define_ability_methods(type_definition);
        }
    }

    /// Emit a shared type's `release_T` function: `release(p) = if RcDecrement(p) then { <release
    /// pointee's shared fields>; FreeShared(p) }`.  The glue is built directly in MIR from the
    /// type's structure -- it cannot be synthesized in the frontend, which would need this type's
    /// own (in-progress) `TypeCheck` result. Nested shared fields become recursive `release_U`
    /// calls (terminating -- calls, not inline); non- shared owned fields are left un-dropped for
    /// now.  Generic over the type's parameters so monomorphization specializes it per element
    /// type, reached only through the `Instantiate` a release call site emits.
    fn define_release_function(&mut self, type_definition: &cst::TypeDefinition) {
        // Recover the type's generics so recursive-release `Instantiate` bindings map to the right
        // MIR generics (the last constructor already left them in scope, but be explicit).
        let tc_generics: Vec<_> = type_definition
            .generics
            .iter()
            .map(|p| type_inference::generics::Generic::Named(Origin::Local(p.name)))
            .collect();
        if tc_generics.is_empty() {
            self.generics_in_scope.clear();
        } else {
            let forall = TCType::Forall(Arc::new(tc_generics), Arc::new(TCType::UNIT));
            self.set_generics_in_scope(&forall);
        }
        let generic_count = self.generics_in_scope.len() as u32;

        // The shared type applied to its own generics -- the TC type of the handle, used to read the
        // pointee layout and to name recursive releases. `generic_args` (the same generics as
        // in-scope MIR generics) are also passed to `type_body` so it substitutes the type's
        // parameters with *these* -- passing `None` would instantiate them as fresh, unbound type
        // variables that convert to `Type::Error`.
        let type_name = TopLevelName::new(self.top_level_id, type_definition.name);
        let generic_args = mapvec(&type_definition.generics, |p| {
            TCType::Generic(type_inference::generics::Generic::Named(Origin::Local(p.name)))
        });
        let mut self_tc = TCType::UserDefined(Origin::TopLevelDefinition(type_name));
        if !generic_args.is_empty() {
            self_tc = TCType::Application(Arc::new(self_tc), Arc::new(generic_args.clone()));
        }

        let name: Name = Arc::new(format!("release_{}", self.context()[type_definition.name].as_ref()));
        // The handle is a shared pointer regardless of the element type, so the signature is
        // `fn(Pointer) -> Unit`; genericity lives in `generic_count` + the body's Instantiates.
        let fn_type = Type::Function(Arc::new(crate::mir::FunctionType {
            parameters: vec![Type::POINTER],
            environment: Type::NO_CLOSURE_ENV,
            return_type: Type::UNIT,
        }));

        let old_scope = std::mem::take(&mut self.local_variables);
        let old_mutables = std::mem::take(&mut self.mutable_locals);

        let id = self.new_definition(name, Some(NameId::RELEASE_FUNCTION), generic_count, fn_type, |this| {
            this.push_parameter(Type::POINTER);
            let handle = Value::Parameter(this.current_block, 0);

            let reached_zero = this.push_instruction(Instruction::RcDecrement(handle), Type::BOOL);
            let glue_block = this.push_block_no_params();
            let cont = this.push_block_no_params();
            this.terminate_block(TerminatorInstruction::if_(reached_zero, glue_block, cont, cont));

            // Last reference: release the pointee's shared fields, then free the whole block.
            this.switch_to_block(glue_block);
            this.build_release_glue(handle, &self_tc, &generic_args);
            this.push_instruction(Instruction::FreeShared(handle), Type::UNIT);
            this.terminate_block(TerminatorInstruction::jmp_no_args(cont));

            this.switch_to_block(cont);
            this.terminate_block(TerminatorInstruction::Return(Value::Unit));
        });

        self.local_variables = old_scope;
        self.mutable_locals = old_mutables;
        self.name_to_id.insert(TopLevelName::new(self.top_level_id, NameId::RELEASE_FUNCTION), id);
    }

    /// Emit the pointee-field releases for a shared type's `release_T`. Derefs the handle to the
    /// pointee layout, then walks it via [Self::release_inline_body]. Leaves the builder positioned
    /// at a single continuation block.
    fn build_release_glue(&mut self, handle: Value, self_tc: &TCType, generic_args: &[TCType]) {
        let inner = self.deref_if_shared(handle, self_tc);
        // The pointee is processed as an inline aggregate of *this* type -- but not through
        // [Self::release_inline_value], which would re-enter `release_T` recursively (a self-call).
        self.release_inline_body(inner, self.top_level_id, generic_args);
    }

    /// Release every shared handle reachable from `value`, an inline aggregate value laid out as the
    /// body of user type `type_id` applied to `args`. For a product this releases each field in
    /// place; for a sum it switches on the tag and releases the active variant's payloads, leaving
    /// the builder at a single merge block. Directly-shared fields become recursive `release_T`
    /// calls; nested non-shared aggregates (`Maybe (shared T)`, tuples, …) are traversed inline so
    /// the shared values they wrap are still released. Bare generics are skipped (v1 leak).
    fn release_inline_body(&mut self, value: Value, type_id: TopLevelId, args: &[TCType]) {
        // Substitute the type's parameters with these args; `None` would instantiate them as fresh
        // unbound type variables (→ `Type::Error` at codegen).
        let args = (!args.is_empty()).then_some(args);
        match type_id.type_body(args, self.compiler) {
            crate::type_inference::TypeBody::Product { fields, .. } => {
                let variant = self.extract_variant(value, 0);
                for (i, (_, field_tc)) in fields.iter().enumerate() {
                    self.release_inline_field(variant, i as u32, &field_tc);
                }
            },
            crate::type_inference::TypeBody::Sum(variants) => {
                let tag = self.extract_tag_value(value);
                // `else_`, the per-variant cases, and the merge `end` must all be distinct blocks
                // (the topological sort treats `end` as the merge of the others).
                let else_block = self.push_block_no_params();
                let end = self.push_block_no_params();
                let case_blocks: Vec<(u32, (BlockId, Option<Value>))> =
                    (0..variants.len()).map(|i| (i as u32, (self.push_block_no_params(), None))).collect();
                self.terminate_block(TerminatorInstruction::Switch {
                    int_value: tag,
                    cases: case_blocks.clone(),
                    else_: (else_block, None),
                    end,
                });
                // All tags are covered; the else edge is unreachable but must be well-formed.
                self.switch_to_block(else_block);
                self.terminate_block(TerminatorInstruction::jmp_no_args(end));
                for (idx, (_, payloads)) in variants.iter().enumerate() {
                    self.switch_to_block(case_blocks[idx].1.0);
                    if !payloads.is_empty() {
                        let variant = self.extract_variant(value, idx);
                        for (j, payload_tc) in payloads.iter().enumerate() {
                            self.release_inline_field(variant, j as u32, payload_tc);
                        }
                    }
                    self.terminate_block(TerminatorInstruction::jmp_no_args(end));
                }
                self.switch_to_block(end);
            },
        }
    }

    /// Project field `index` out of `tuple` and release the shared handles reachable from it.
    fn release_inline_field(&mut self, tuple: Value, index: u32, field_tc: &TCType) {
        if self.shared_release_target(field_tc).is_none() && self.aggregate_type_target(field_tc).is_none() {
            return; // primitives and bare generics carry no shared handle to release.
        }
        let field_type = self.convert_type(field_tc, None);
        let field = self.push_instruction(Instruction::IndexTuple { tuple, index }, field_type);
        self.release_inline_value(field, field_tc);
    }

    /// Release the shared handles reachable from `value` of type `tc`: a directly-shared handle
    /// becomes a recursive `release_T` call; a non-shared aggregate is traversed inline via
    /// [Self::release_inline_body]; anything else carries no shared handle and is skipped.
    fn release_inline_value(&mut self, value: Value, tc: &TCType) {
        if let Some((type_id, type_args)) = self.shared_release_target(tc) {
            self.emit_release_call(value, type_id, &type_args);
        } else if let Some((type_id, type_args)) = self.aggregate_type_target(tc) {
            self.release_inline_body(value, type_id, &type_args);
        }
    }

    /// Emit `release_T(handle)` for shared type `type_id` applied to `type_args` (instantiating the
    /// generic release function when the type is generic).
    fn emit_release_call(&mut self, handle: Value, type_id: TopLevelId, type_args: &[TCType]) {
        let release_name = TopLevelName::new(type_id, NameId::RELEASE_FUNCTION);
        let release_id = self.get_definition_id(&release_name);
        let fn_type = Type::Function(Arc::new(crate::mir::FunctionType {
            parameters: vec![Type::POINTER],
            environment: Type::NO_CLOSURE_ENV,
            return_type: Type::UNIT,
        }));
        let callee = if type_args.is_empty() {
            self.make_definition_value(release_id, Arc::new("release".to_string()), fn_type)
        } else {
            let mir_args = mapvec(type_args, |a| self.convert_type(a, None));
            self.push_instruction(Instruction::Instantiate(release_id, Arc::new(mir_args)), fn_type)
        };
        self.push_instruction(Instruction::Call { function: callee, arguments: vec![handle] }, Type::UNIT);
    }

    /// If `typ` resolves to a *non-shared* user-defined type (a product/sum whose inline layout may
    /// still wrap shared handles), return its item id and type arguments so its body can be walked.
    /// Shared types return `None` here -- [Self::shared_release_target] handles those.
    fn aggregate_type_target(&self, typ: &TCType) -> Option<(TopLevelId, Vec<TCType>)> {
        if self.shared_release_target(typ).is_some() {
            return None;
        }
        match typ.follow(&self.types.bindings) {
            TCType::Application(constructor, args) => match constructor.follow(&self.types.bindings) {
                TCType::UserDefined(Origin::TopLevelDefinition(name)) => Some((name.top_level_item, args.to_vec())),
                _ => None,
            },
            TCType::UserDefined(Origin::TopLevelDefinition(name)) => Some((name.top_level_item, Vec::new())),
            _ => None,
        }
    }

    /// If `typ` resolves to a `shared` user-defined type, return its item id and type arguments --
    /// enough to name and (if generic) instantiate that type's `release_T` for a recursive call.
    fn shared_release_target(&self, typ: &TCType) -> Option<(TopLevelId, Vec<TCType>)> {
        let is_shared = |name: &TopLevelName| {
            let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
            matches!(&item.kind, cst::TopLevelItemKind::TypeDefinition(td) if td.shared)
        };
        match typ.follow(&self.types.bindings) {
            TCType::Application(constructor, args) => match constructor.follow(&self.types.bindings) {
                TCType::UserDefined(Origin::TopLevelDefinition(name)) if is_shared(name) => {
                    Some((name.top_level_item, args.to_vec()))
                },
                _ => None,
            },
            TCType::UserDefined(Origin::TopLevelDefinition(name)) if is_shared(name) => {
                Some((name.top_level_item, Vec::new()))
            },
            _ => None,
        }
    }

    fn define_type_constructor(
        &mut self, name_id: NameId, constructor_type: &TCType,
        field_types: &[crate::type_inference::types::ParameterType], tag: Option<u8>, shared: bool,
    ) {
        let top_level_name = TopLevelName::new(self.top_level_id, name_id);
        let name = self.context()[name_id].clone();
        let typ = self.convert_type(constructor_type, None);

        // The unboxed layout constructor body builds. For non-shared types this is the same as
        // the final type. For shared types, we wrap this in a pointer at the end.
        let payload_type = if shared {
            let return_typ =
                constructor_type.ignore_forall().return_type().unwrap_or_else(|| constructor_type.ignore_forall());
            self.shared_inner_layout_of(return_typ).unwrap_or_else(|| {
                let name = self.context()[name_id].clone();
                panic!("shared constructor {name} has no inner layout")
            })
        } else {
            typ.function_return_type().cloned().unwrap_or_else(|| typ.clone())
        };

        let raw_union_type = payload_type.without_union_tag();
        let generic_count = self.generics_in_scope.len() as u32;
        let is_zero_arg = field_types.is_empty();

        let id = self.new_definition(name, Some(name_id), generic_count, typ, |this| {
            let mut result = this.build_constructor_payload(field_types, tag, &payload_type, raw_union_type, name_id);
            if shared {
                result = this.push_instruction(Instruction::AllocShared(result), Type::POINTER);
            }

            // 0-arg constructors are globals (`Result` terminator → `is_global()` is true).
            // For shared 0-arg constructors the AllocShared lowers to a backing static in
            // the constant codegen path.
            let terminator = if is_zero_arg { TerminatorInstruction::Result } else { TerminatorInstruction::Return };
            this.terminate_block(terminator(result));
        });
        self.name_to_id.insert(top_level_name, id);
    }

    /// Build the value a constructor returns *before* any shared-pointer wrap.
    /// Materializes block parameters, packs them into a tuple, and (for sum-type
    /// variants) transmutes & wraps `(tag, union)`.
    fn build_constructor_payload(
        &mut self, field_types: &[crate::type_inference::types::ParameterType], tag: Option<u8>, payload_type: &Type,
        raw_union_type: Option<Type>, name_id: NameId,
    ) -> Value {
        let field_types = mapvec(field_types, |param| self.convert_type(&param.typ, None));
        let fields = mapvec(field_types.iter().enumerate(), |(i, field_type)| {
            self.push_parameter(field_type.clone());
            Value::Parameter(BlockId::ENTRY_BLOCK, i as u32)
        });

        let mut payload = self.push_instruction(Instruction::MakeTuple(fields), Type::tuple(field_types));

        if let Some(tag) = tag {
            let raw_union_type = raw_union_type.unwrap_or_else(|| {
                let name = self.context()[name_id].clone();
                panic!("Failed to unwrap raw union type. Full result type is: {payload_type} for constructor {name}")
            });
            let casted = self.push_instruction(Instruction::Transmute(payload), raw_union_type);
            payload = self
                .push_instruction(Instruction::MakeTuple(vec![Value::tag_value(tag), casted]), payload_type.clone());
        }

        payload
    }

    fn define_ability_methods(&mut self, type_definition: &cst::TypeDefinition) {
        if let cst::TypeDefinitionBody::Struct(fields) = &type_definition.body {
            let constructor_type = self.types.get_generalized(type_definition.name);
            self.set_generics_in_scope(constructor_type);
            let generic_count = self.generics_in_scope.len() as u32;
            let constructor_mir_type = self.convert_type(constructor_type, None);

            let struct_type =
                constructor_mir_type.function_return_type().cloned().unwrap_or_else(|| constructor_mir_type.clone());

            let field_mir_types: Vec<Type> =
                if let Type::Function(fn_type) = &constructor_mir_type { fn_type.parameters.clone() } else { vec![] };

            for (i, (field_name_id, _)) in fields.iter().enumerate() {
                let Some(field_type) = field_mir_types.get(i) else { continue };

                // Only generate wrappers for function-typed fields (all ability methods).
                // TODO: We should still generate wrappers for other types
                let Type::Function(fn_type) = field_type else { continue };
                let (value_param_types, return_type) = (fn_type.parameters.clone(), fn_type.return_type.clone());

                let mut wrapper_params = value_param_types;
                wrapper_params.push(struct_type.clone());
                let wrapper_type = Type::Function(Arc::new(super::FunctionType {
                    parameters: wrapper_params.clone(),
                    environment: Type::NO_CLOSURE_ENV,
                    return_type: return_type.clone(),
                }));

                let name = self.context()[*field_name_id].clone();
                let top_level_name = TopLevelName::new(self.top_level_id, *field_name_id);
                let field_type_clone = field_type.clone();
                let field_is_closure = field_type_clone.is_closure();
                let n_params = wrapper_params.len();

                let id = self.new_definition(name.clone(), Some(*field_name_id), generic_count, wrapper_type, |this| {
                    for pt in &wrapper_params {
                        this.push_parameter(pt.clone());
                    }
                    // Last parameter is the implicit struct receiver.
                    let struct_param = Value::Parameter(BlockId::ENTRY_BLOCK, n_params as u32 - 1);
                    let extracted = this.push_instruction(
                        Instruction::IndexTuple { tuple: struct_param, index: i as u32 },
                        field_type_clone,
                    );
                    let value_args = mapvec(0..n_params - 1, |j| Value::Parameter(BlockId::ENTRY_BLOCK, j as u32));
                    let instruction = if field_is_closure {
                        Instruction::CallClosure { closure: extracted, arguments: value_args }
                    } else {
                        Instruction::Call { function: extracted, arguments: value_args }
                    };
                    let result = this.push_instruction(instruction, return_type.clone());
                    this.terminate_block(TerminatorInstruction::Return(result));
                });
                self.name_to_id.insert(top_level_name, id);
            }
        }
    }
}
