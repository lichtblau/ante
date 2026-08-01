//! Translates a single top-level item into MIR, in parallel with other items. See [super] for MIR itself.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, LazyLock},
};

use dashmap::DashMap;
use inc_complete::DbGet;
use rustc_hash::FxHashMap;

use crate::{
    incremental::{ExportedTypes, GetItem, GetItemRaw, TypeCheck},
    iterator_extensions::mapvec,
    lexer::token::{FloatKind, Integer, IntegerKind},
    mir::{
        Block, BlockId, Definition, DefinitionId, FloatConstant, FunctionType, Generic, Instruction, IntConstant, Mir,
        TerminatorInstruction, Type, Value, next_definition_id,
    },
    name_resolution::{Origin, namespace::SourceFileId},
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

/// Shared between concurrent calls of [build_initial_mir].
static NAME_IDS: LazyLock<SharedIdsMap> = LazyLock::new(DashMap::new);

/// Look up a MIR function by a [TopLevelName]
pub(crate) fn lookup_definition_id(name: &TopLevelName) -> Option<DefinitionId> {
    NAME_IDS.get(name).map(|entry| *entry.value())
}

/// Builds the MIR with the default shared global [SharedIdsMap].
pub(crate) fn build_initial_mir_with_shared_map<T>(compiler: &T, item_id: TopLevelId) -> Option<Mir>
where
    T: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw> + DbGet<ExportedTypes>,
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
    T: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw> + DbGet<ExportedTypes>,
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
        cst::TopLevelItemKind::TraitDefinition(_) | cst::TopLevelItemKind::EffectDefinition(_) => {
            unreachable!("Traits/effects should be desugared to types")
        },
        cst::TopLevelItemKind::TraitImpl(_) => unreachable!("TraitImpls should be desugared to definitions"),
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

/// Whether a top-level item is a trait, an effect, or neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbilityKind {
    NotAbility,
    Trait,
    Effect,
}

/// Where a call's evidence for one concrete effect comes from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapabilitySource {
    CanClause(usize),
    Handle(ExprId),
}

/// Saved/restored around building a definition in isolation. See [Context::take_scope].
struct SavedScope {
    local_variables: FxHashMap<NameId, Value>,
    mutable_locals: rustc_hash::FxHashSet<NameId>,
    own_row: Vec<(TCType, Value)>,
    handle_capabilities: FxHashMap<ExprId, (TCType, Value)>,
    capability_bundle: Option<Value>,
}

/// The per-[TopLevelId] context. Intended for each top-level item convert to MIR in parallel.
struct Context<'local, Db> {
    compiler: &'local Db,
    types: &'local TypeCheckResult,

    top_level_id: TopLevelId,

    generics_in_scope: FxHashMap<type_inference::generics::Generic, Generic>,

    current_function: Option<Definition>,
    current_block: BlockId,

    global_variables: FxHashMap<Origin, Value>,
    local_variables: FxHashMap<NameId, Value>,

    /// Names of local mutable variables. These have a StackAlloc'd pointer as their MIR value.
    mutable_locals: rustc_hash::FxHashSet<NameId>,

    /// Stack of `(continue_target, break_target)` for each enclosing `while`/`for` loop.
    /// Innermost loop is last. Swapped out across lambda boundaries.
    loop_targets: Vec<(BlockId, BlockId)>,

    finished_functions: FxHashMap<DefinitionId, Definition>,
    name_to_id: &'local SharedIdsMap,

    /// Any external items will have their name & type stored here
    external: FxHashMap<DefinitionId, super::Extern>,

    /// Cache of whether each item is a trait/effect or not. Caching this avoids reissuing GetItemRaw per call site.
    ability_defs: FxHashMap<TopLevelId, AbilityKind>,

    /// The Prelude's `Extract` ability (the item behind `(.[])`), looked up once. `None` once
    /// resolved-and-absent; the outer `Option` is the "not looked up yet" state. See
    /// [Context::path_is_extract_method].
    extract_ability: Option<Option<TopLevelId>>,

    /// Position of each effect op within its ability's body. Propagated to [Mir::preserved_op_indices]
    /// so [crate::mir::effects::effect_lowering] can look up the slot of an op in the capability tuple.
    effect_op_indices: FxHashMap<DefinitionId, u32>,

    /// The current function's effect row: (effect type, capability value) pairs in evidence-tuple order.
    effects: Vec<(TCType, Value)>,

    /// Capabilities from enclosing `handle`s, keyed by the handle's [ExprId].
    handle_capabilities: FxHashMap<ExprId, (TCType, Value)>,

    /// This function's own capability-bundle parameter for its open effect-row tail, if any.
    capability_bundle: Option<Value>,
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
            extract_ability: Default::default(),
            effect_op_indices: Default::default(),
            effects: Default::default(),
            handle_capabilities: Default::default(),
            capability_bundle: None,
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
    Db: DbGet<TypeCheck> + DbGet<GetItem> + DbGet<GetItemRaw> + DbGet<ExportedTypes>,
{
    fn push_instruction(&mut self, instruction: Instruction, result_type: Type) -> Value {
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

    /// Pushes a parameter and returns the `Value` referring to it.
    fn push_capability_parameter(&mut self, parameter_type: Type) -> Value {
        self.push_parameter(parameter_type);
        let index = self.current_block().parameter_types.len() as u32 - 1;
        Value::Parameter(self.current_block, index)
    }

    /// Pushes a capability parameter for each of `parameters`, in order.
    fn push_capability_parameters(&mut self, parameters: &[Type]) -> Vec<Value> {
        mapvec(parameters, |typ| self.push_capability_parameter(typ.clone()))
    }

    /// Panics if the shapes are incompatible, since evidence shapes should always nest.
    fn project_evidence_or_panic(&mut self, provided: Value, expected: &Type, needed: &Type) -> Value {
        self.project_evidence(provided, expected, needed).unwrap_or_else(|| {
            panic!("cannot project argument evidence {needed} out of the expected evidence {expected}")
        })
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
                    // Defaults to F64 if the type variable is still unbound
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
        // An effect operation as a first-class value gets its capability through its own evidence.
        if let Some((effect_op, op_index, AbilityKind::Effect)) = self.try_resolve_ability_method(path_id) {
            self.effect_op_indices.insert(effect_op, op_index);
            return self.effect_op_value_wrapper(path_id, op_index);
        }

        // A synthesized `release_T` is C-shaped (see [Self::release_function_mir_type]), so its
        // type never comes from `path_types`.
        let is_release_function = matches!(
            self.context().path_origin(path_id),
            Some(Origin::TopLevelDefinition(name)) if name.local_name_id == NameId::RELEASE_FUNCTION
        );

        // Deliberately allow us to reference variables not in the context.
        // This allows us to convert all definitions to MIR in parallel, trusting
        // that the links will work out later.
        let mut value =
            match self.context().path_origin(path_id) {
                Some(Origin::TopLevelDefinition(name)) => {
                    let id = self.get_definition_id(&name);
                    let is_extern = self.name_is_extern(&name);
                    let name = self.get_definition_name(&name);
                    if is_release_function {
                        self.make_definition_value(id, name, Self::release_function_mir_type())
                    } else if is_extern {
                        let tc_type = &self.types.result.maps.path_types[&path_id];
                        let c_type = self.convert_context().convert_c_function_type(tc_type);
                        let target = self.make_definition_value(id, name.clone(), c_type);
                        self.extern_evidence_wrapper(target, name, self.convert_path_type(path_id))
                    } else {
                        let typ = self.convert_path_type(path_id);
                        self.make_definition_value(id, name, typ)
                    }
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
            let typ =
                if is_release_function { Self::release_function_mir_type() } else { self.convert_path_type(path_id) };
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
            let typ = self.types.get_generalized(name_id.unwrap());
            self.set_generics_in_scope(&typ);
        }

        let previous_state = self.is_non_function_global(definition).then(|| {
            let generic_count = self.generics_in_scope.len() as u32;
            // Extern globals hold C-shaped function values: no evidence parameter.
            let pattern_type = &self.types.result.maps.pattern_types[&definition.pattern];
            let typ = match self.rhs_is_extern(definition.rhs) {
                true => self.convert_context().convert_c_function_type(pattern_type),
                false => self.convert_pattern_type(definition.pattern),
            };
            self.start_global(name, name_id, generic_count, typ)
        });

        // `Copy` on a `shared` type is the compiler's to define: see [Self::build_shared_copy_impl].
        let shared_copy = if is_global && name_id.is_some_and(|name_id| self.is_shared_copy_impl(name_id)) {
            let impl_type = self.convert_pattern_type(definition.pattern);
            self.build_shared_copy_impl(impl_type)
        } else {
            None
        };

        let value = match shared_copy {
            Some(value) => value,
            None => match &self.context()[definition.rhs] {
                cst::Expr::Lambda(lambda) => {
                    let name = self.try_find_name(definition.pattern).map(|(name, _)| name);
                    self.lambda(lambda, name_id, name, definition.rhs, is_global)
                },
                _ => self.expression(definition.rhs),
            },
        };

        self.bind_pattern(definition.pattern, value);
        // Binding from an existing shared handle creates a new owning location; the binding's
        // scope-exit release balances this retain. Only real drop-registered bindings are
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

        // If the object has a reference/pointer type, the MIR value is a pointer, so use GetFieldPtr instead.
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

    /// Retain (bump the refcount of) `value` when `expr` is a shared or heap-env-closure place --
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
    /// pointer (`IndexTuple(closure, 1)` + null-safe `RetainClosureEnv`). An aggregate that holds
    /// either one is retained through, field by field ([Self::retain_inline_value]). Everything else
    /// is a no-op. The caller has already established `expr` is a place worth retaining.
    fn emit_place_retain(&mut self, expr: ExprId, value: Value) {
        if self.expr_tc_is_shared(expr) {
            self.push_instruction(Instruction::RcRetain(value), Type::UNIT);
        } else if self.expr_tc_is_heap_env_closure(expr) {
            let env = self.push_instruction(Instruction::IndexTuple { tuple: value, index: 1 }, Type::POINTER);
            self.push_instruction(Instruction::RetainClosureEnv(env), Type::UNIT);
        } else if self.context().is_copy_place(expr) {
            // `Maybe (Node b)` is not a handle, so neither branch above fires -- but its drop
            // releases the handle inside it, and the frontend marked this read a copy, meaning the
            // source it came from will run that same drop too. Retain exactly what both will
            // release. A move is never marked (the obligation left with the value), and a place with
            // no refcounted data anywhere inside it is not marked either, so the common read of an
            // ordinary aggregate does not even reach the walk.
            let typ = self.types.result.maps.expr_types[&expr].follow(&self.types.bindings).clone();
            self.retain_inline_value(value, &typ);
        }
    }

    /// Retain every refcounted value `value` holds inline, for a `value` of type `tc` that was
    /// copied out of a place: a shared handle is bumped, a heap-env closure's environment is bumped,
    /// a product is walked field by field, and a sum switches on its tag so only the active variant's
    /// payloads are retained (leaving the builder at a single merge block).
    ///
    /// This is the retain side of [Self::try_synthesize_drop_for_place]'s structural expansion -- the
    /// drop the copy itself will run -- and the two must stay symmetric: a handle this walk misses is
    /// released twice (a use-after-free), one it invents is never released (a leak). It stops where
    /// that drop stops: at a `Ptr`/`ref` (which owns nothing), and at a handle (a retain bumps the
    /// cell it names; only a release at count zero recurses into the pointee).
    fn retain_inline_value(&mut self, value: Value, tc: &TCType) {
        if !self.type_owns_refcount_inline(tc) {
            return;
        }
        if self.shared_release_target(tc).is_some() {
            self.push_instruction(Instruction::RcRetain(value), Type::UNIT);
        } else if self.tc_is_heap_env_closure(tc) {
            let env = self.push_instruction(Instruction::IndexTuple { tuple: value, index: 1 }, Type::POINTER);
            self.push_instruction(Instruction::RetainClosureEnv(env), Type::UNIT);
        } else if let TCType::Tuple(elements) = tc.follow(&self.types.bindings) {
            for (i, element) in elements.clone().iter().enumerate() {
                self.retain_inline_field(value, i as u32, element);
            }
        } else if let Some((type_id, type_args)) = self.aggregate_type_target(tc) {
            self.retain_inline_body(value, type_id, &type_args);
        }
    }

    /// Project field `index` out of `tuple` and retain the refcounted values reachable from it.
    /// Fields that hold none are skipped without projecting: no instruction is emitted for them.
    fn retain_inline_field(&mut self, tuple: Value, index: u32, field_tc: &TCType) {
        if !self.type_owns_refcount_inline(field_tc) {
            return;
        }
        let field_type = self.convert_type(field_tc, None);
        let field = self.push_instruction(Instruction::IndexTuple { tuple, index }, field_type);
        self.retain_inline_value(field, field_tc);
    }

    /// Retain the refcounted values held inline by `value`, an aggregate laid out as the body of user
    /// type `type_id` applied to `args`. Mirrors [Self::release_inline_body] block for block, minus
    /// its tail-slot machinery: a retain never cascades into a pointee, so there is no self-recursive
    /// chain to unroll into a loop.
    fn retain_inline_body(&mut self, value: Value, type_id: TopLevelId, args: &[TCType]) {
        // Substitute the type's parameters with these args; `None` would instantiate them as fresh
        // unbound type variables (→ `Type::Error` at codegen).
        let args_opt = (!args.is_empty()).then_some(args);
        match type_id.type_body(args_opt, self.compiler) {
            crate::type_inference::TypeBody::Product { fields, .. } => {
                let variant = self.extract_variant(value, 0);
                for (i, (_, field_tc)) in fields.iter().enumerate() {
                    self.retain_inline_field(variant, i as u32, field_tc);
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
                            self.retain_inline_field(variant, j as u32, payload_tc);
                        }
                    }
                    self.terminate_block(TerminatorInstruction::jmp_no_args(end));
                }
                self.switch_to_block(end);
            },
        }
    }

    /// Does a value of this type hold reference-counted data inline -- a shared handle or a
    /// heap-env closure, reachable by walking fields and variant payloads? Asked before a retain walk
    /// emits anything, so copying an ordinary aggregate costs nothing (not even a tag switch).
    ///
    /// The walk terminates for the same reason [Self::release_inline_body]'s does: it stops at
    /// handles and at `Ptr`/`ref`, and an aggregate that contained itself inline through neither
    /// would have no finite layout to begin with.
    fn type_owns_refcount_inline(&self, tc: &TCType) -> bool {
        if self.shared_release_target(tc).is_some() || self.tc_is_heap_env_closure(tc) {
            return true;
        }
        if let TCType::Tuple(elements) = tc.follow(&self.types.bindings) {
            return elements.iter().any(|element| self.type_owns_refcount_inline(element));
        }
        let Some((type_id, args)) = self.aggregate_type_target(tc) else { return false };
        let args_opt = (!args.is_empty()).then_some(args.as_slice());
        match type_id.type_body(args_opt, self.compiler) {
            crate::type_inference::TypeBody::Product { fields, .. } => {
                fields.iter().any(|(_, field)| self.type_owns_refcount_inline(field))
            },
            crate::type_inference::TypeBody::Sum(variants) => variants
                .iter()
                .any(|(_, payloads)| payloads.iter().any(|payload| self.type_owns_refcount_inline(payload))),
        }
    }

    /// The Prelude's `Copy` ability, if the Prelude declares one.
    fn copy_ability_name(&self) -> Option<TopLevelName> {
        ExportedTypes(SourceFileId::prelude()).get(self.compiler).get(&Arc::new("Copy".to_string())).copied()
    }

    /// True when `path` names a method of the Prelude's `Extract` ability (`(.[])`). The ability
    /// item is resolved once and cached: this is asked at every ability-method call site.
    fn path_is_extract_method(&mut self, path: PathId) -> bool {
        let extract = *self.extract_ability.get_or_insert_with(|| {
            ExportedTypes(SourceFileId::prelude())
                .get(self.compiler)
                .get(&Arc::new("Extract".to_string()))
                .map(|name| name.top_level_item)
        });
        let Some(extract) = extract else { return false };
        matches!(self.context().path_origin(path), Some(Origin::TopLevelDefinition(name)) if name.top_level_item == extract)
    }

    /// True when this global definition is an `impl _: Copy S` whose `S` is a `shared` type.
    fn is_shared_copy_impl(&self, name_id: NameId) -> bool {
        // `get_generalized` hands back an owned type, so keep it alive for the borrows below.
        let generalized = self.types.get_generalized(name_id);
        let typ = generalized.ignore_forall().follow(&self.types.bindings);
        let TCType::Application(constructor, arguments) = typ else { return false };

        // Ask the cheap structural question first: `shared_release_target` is local, while the
        // `Copy` lookup queries the prelude. Impls whose target is a shared type are rare.
        if arguments.len() != 1 || self.shared_release_target(&arguments[0]).is_none() {
            return false;
        }
        let TCType::UserDefined(Origin::TopLevelDefinition(ability)) = constructor.follow(&self.types.bindings) else {
            return false;
        };
        Some(*ability) == self.copy_ability_name()
    }

    /// Build the value of an `impl _: Copy S` for a `shared` type `S`, in place of the user's body.
    ///
    /// `Copy`'s `(.*): fn (ref t) -> t` has to hand back an owned (+1) handle: that is the
    /// convention every other producer of a shared handle already follows (a `Constructor` allocates
    /// at count 1; a function returning a handle retains it), and it is what lets every consumer
    /// release what it was given. The only body the surface language lets the user write is `deref`
    /// -- a bare load, which aliases at +0 -- and it exposes no retain primitive to fix that with.
    /// So the release runs unmatched and frees a cell that is still reachable. The compiler
    /// therefore supplies this method itself: load the handle, then retain it.
    ///
    /// This is the one place both shapes bottom out. A borrowed field read (`h.head` behind a
    /// `ref`/`mut`) is coerced by the frontend into a `(.*)` call, and a generic `{Copy t}` caller
    /// dispatches through this same impl -- and only the impl can serve the generic case, since
    /// inside such a caller `t` is still abstract and no call-site retain could know to fire.
    ///
    /// Returns `None` if the impl does not have the expected one-method shape, leaving the user's
    /// body to be lowered as written.
    fn build_shared_copy_impl(&mut self, impl_type: Type) -> Option<Value> {
        let Type::Tuple(fields) = &impl_type else { return None };
        let [Type::Function(method)] = fields.as_slice() else { return None };
        // A shared handle is a pointer; `RcRetain` requires one.
        if method.return_type != Type::POINTER {
            return None;
        }

        // The ability's field holds a closure. The function behind it takes the closure's
        // environment as a trailing parameter -- the shape `coerce_to_closure` packs.
        let closure_type = Type::Function(method.clone());
        let return_type = method.return_type.clone();
        let mut parameters = method.parameters.clone();
        parameters.push(method.environment.clone());

        let function_type = Type::Function(Arc::new(crate::mir::FunctionType {
            parameters: parameters.clone(),
            environment: Type::NO_CLOSURE_ENV,
            return_type: return_type.clone(),
        }));

        let name: Name = Arc::new("copy_shared".to_string());
        let generic_count = self.generics_in_scope.len() as u32;
        let id = self.new_definition(name.clone(), None, generic_count, function_type.clone(), |this| {
            for parameter in &parameters {
                this.push_parameter(parameter.clone());
            }
            let reference = Value::Parameter(BlockId::ENTRY_BLOCK, 0);
            let handle = this.push_instruction(Instruction::Deref(reference), return_type);
            this.push_instruction(Instruction::RcRetain(handle), Type::UNIT);
            this.terminate_block(TerminatorInstruction::Return(handle));
        });

        let method = self.make_definition_value(id, name, function_type);
        let environment = self.push_instruction(Instruction::Transmute(Value::Unit), Type::POINTER);
        let closure = self.push_instruction(Instruction::PackClosure { function: method, environment }, closure_type);
        Some(self.push_instruction(Instruction::MakeTuple(vec![closure]), impl_type))
    }

    fn call(&mut self, call: &cst::Call, id: ExprId) -> Value {
        // Intrinsic calls must lower to concrete instructions here, not a function wrapper.
        if let Some(result) = self.try_lower_intrinsic(call, id) {
            return result;
        }

        let diverges = self.callee_diverges(call.function);
        let result_type = if diverges { Type::UNIT } else { self.expr_type(id) };

        // Trait & effect method calls dispatch through `IndexTuple cap op_index + CallClosure`.
        if let cst::Expr::Variable(path_id) = &self.context()[call.function]
            && let Some((effect_op, op_index, kind)) = self.try_resolve_ability_method(*path_id)
        {
            // A capability built by applying a constrained impl (`drop_vec drop_i32`) heap-
            // allocates each method closure's environment inside the impl function, freshly
            // per call site execution. A `Drop` capability cannot outlive its method call
            // (unlike e.g. Stream combinators, which legitimately capture their caps in
            // returned closures), so for Drop calls we lower application-shaped capability
            // arguments through `lower_drop_capability` -- collecting every nesting level's
            // environments -- and free them once the drop completes. Static impls and
            // capability parameters lower as plain variables and are never freed.
            //
            // Only a trait call can be a Drop call: effect capabilities are not passed as
            // arguments at all, the callee borrows them from its evidence parameter.
            let is_drop_call = matches!(kind, AbilityKind::Trait)
                && matches!(
                    self.context().path_origin(*path_id),
                    Some(Origin::TopLevelDefinition(name)) if self.is_prelude_drop_trait(name.top_level_item)
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

            let result = match kind {
                AbilityKind::Trait => {
                    self.emit_trait_method_call(call.function, effect_op, op_index, arguments, result_type, diverges)
                },
                AbilityKind::Effect => {
                    self.emit_effect_op_call(call.function, effect_op, op_index, arguments, result_type, diverges)
                },
                AbilityKind::NotAbility => unreachable!(),
            };
            if !diverges {
                for environment in cap_env_frees {
                    self.emit_free(environment);
                }
                // `Extract`'s `(.[])` is a read: the collection still holds the element it hands
                // back, so an element that owns refcounted data comes back as an alias while the
                // caller drops it as an owned value. The prelude's `Ptr` impl is a bare `deref_ptr`
                // and says so itself ("TODO: Remove. This breaks ownership"); nothing in the impl
                // can fix it, because there the element type is a bare generic. The call site is
                // where the element type is finally concrete, so it is where the copy is funded --
                // the same rule as a copied field read, for the one place-like read that lowers to
                // a call. Impls returning `ref`/`mut` elements retain nothing (a borrow owns
                // nothing), and inside generic code the element stays opaque and is skipped, so a
                // `Vec` index that delegates to the `Ptr` impl is funded exactly once, out here.
                if self.path_is_extract_method(*path_id) {
                    let typ = self.types.result.maps.expr_types[&id].follow(&self.types.bindings).clone();
                    self.retain_inline_value(result, &typ);
                }
            }
            return result;
        }

        let function = self.expression(call.function);
        // A synthesized `release_T` call consumes a reference (it decrements), so its argument
        // must not be retained -- retaining would cancel the decrement and leak. Every other call is
        // callee-owns: a shared handle passed by value is a new owning location, so retain it (the
        // callee's parameter release, or the constructed value's later release, balances it).
        let is_release_call = matches!(&self.context()[call.function], cst::Expr::Variable(path)
            if matches!(self.context().path_origin(*path), Some(Origin::TopLevelDefinition(name)) if name.local_name_id == NameId::RELEASE_FUNCTION));
        // A statically-known callee may have borrowing parameters (it never releases them). For a
        // shared place argument in such a position, the callee-owns retain is elided outright
        // when every implicit argument of this call is an unconstrained static impl (the callee
        // then provably cannot suspend); otherwise the retain stays and is balanced by a post-call
        // release -- the caller's frame holds the handle across any suspension. A shared rvalue
        // argument (fresh, count 1) is caller-owned either way and released after the call.
        let callee_mask = if is_release_call { None } else { self.borrowed_param_mask_of_callee(call.function) };
        let implicits_pure = callee_mask.is_some()
            && call.arguments.iter().filter(|arg| arg.is_implicit).all(|arg| self.implicit_is_pure_static(arg.expr));

        // (value, expr, retain id when this is an elidable pair)
        let mut post_call_releases: Vec<(Value, ExprId, Option<crate::mir::InstructionId>)> = Vec::new();
        let mut implicit_positions: Vec<u32> = Vec::new();
        let mut explicit_index = 0usize;
        let mut arg_index = 0u32;
        let mut arguments = mapvec(&call.arguments, |arg| {
            let value = if arg.is_implicit && !is_release_call {
                implicit_positions.push(arg_index);
                self.expression(arg.expr)
            } else {
                let value = self.expression(arg.expr);
                if !is_release_call {
                    let borrowed = callee_mask
                        .as_ref()
                        .is_some_and(|mask| mask.get(explicit_index).copied().unwrap_or(false))
                        && self.expr_tc_is_shared(arg.expr);
                    if borrowed {
                        let is_place =
                            matches!(&self.context()[arg.expr], cst::Expr::Variable(_) | cst::Expr::MemberAccess(_));
                        if is_place && implicits_pure {
                            // Pair elided: no retain here, no release in the callee.
                        } else if is_place {
                            // Impure implicits: retain + post-call release, recorded as a
                            // borrow pair so the post-mono pass can elide it when the
                            // specialized capability values turn out pure.
                            let retain = self.push_instruction(Instruction::RcRetain(value), Type::UNIT);
                            let retain_id = match retain {
                                Value::InstructionResult(id) => Some(id),
                                _ => None,
                            };
                            post_call_releases.push((value, arg.expr, retain_id));
                        } else {
                            // Rvalue: caller-owned regardless of purity; never elidable.
                            post_call_releases.push((value, arg.expr, None));
                        }
                    } else {
                        self.retain_if_shared_place(arg.expr, value);
                    }
                }
                explicit_index += 1;
                value
            };
            // Adapt the argument's evidence shape last. Every retain, release and borrow-pair
            // record above is keyed to the value the argument expression denotes, and an adapter
            // reuses that value's own environment pointer, so the RC bookkeeping still lands on
            // what the callee receives.
            let value = self.coerce_argument_evidence(value, &function, call.function, arg_index as usize);
            arg_index += 1;
            value
        });
        // The evidence tuple is borrowed data the compiler threads through the call; it is owned
        // by nobody, so it is never retained here and never released by the callee.
        self.append_evidence_argument(call.function, &function, &mut arguments);

        // Every position recorded above -- the borrowing mask's `explicit_index` and the borrow
        // pairs' `implicit_positions` -- indexes the surface arguments. Evidence is appended
        // strictly after them, so those indices stay valid against the emitted argument vector.
        // An off-by-one here is a miscompile with no test signal until refcounts skew.
        debug_assert!(
            implicit_positions.iter().all(|position| (*position as usize) < call.arguments.len()),
            "implicit argument positions must index the pre-evidence argument space"
        );
        debug_assert!(
            arguments.len() == call.arguments.len() || arguments.len() == call.arguments.len() + 1,
            "evidence must be appended after the surface arguments, never interleaved"
        );

        // Coercing argument evidence above may have patched `function`'s own type in place (see `patch_environment_binding`),
        // so re-read the result type off the callee rather than the value computed pre-coercion.
        let function_type = self.type_of_value(&function);
        let result_type = match (&function_type, diverges) {
            (_, true) => Type::UNIT,
            (Type::Function(ft), false) => ft.return_type.clone(),
            _ => result_type,
        };

        let instruction = if function_type.is_closure() {
            Instruction::CallClosure { closure: function, arguments }
        } else {
            Instruction::Call { function, arguments }
        };

        let value = self.push_instruction(instruction, result_type);
        let call_id = match value {
            Value::InstructionResult(id) => Some(id),
            _ => None,
        };
        if diverges {
            self.terminate_block(TerminatorInstruction::Unreachable);
        } else {
            for (arg_value, arg_expr, retain_id) in post_call_releases {
                let release = self.emit_shared_release_for_expr(arg_value, arg_expr);
                if let (Some(retain), Some(call), Some(Value::InstructionResult(release_call))) =
                    (retain_id, call_id, release)
                {
                    let pair = crate::mir::BorrowPair {
                        retain,
                        release_call,
                        call,
                        implicit_args: implicit_positions.clone(),
                    };
                    self.current_function().borrow_pairs.push(pair);
                }
            }
        }
        value
    }

    /// The borrowing-parameter mask of a call's statically-known top-level callee, from its item's
    /// inference result. `None` for indirect calls, effect ops, and callees without borrowing
    /// parameters.
    fn borrowed_param_mask_of_callee(&mut self, function: ExprId) -> Option<Vec<bool>> {
        let cst::Expr::Variable(path) = &self.context()[function] else { return None };
        let Some(Origin::TopLevelDefinition(name)) = self.context().path_origin(*path) else { return None };
        let check = crate::incremental::TypeCheck(name.top_level_item).get(self.compiler);
        let mask = check.result.borrowed_params.get(&name.local_name_id)?;
        (!mask.is_empty()).then(|| mask.clone())
    }

    /// True when an implicit argument is an unconstrained static impl -- a plain reference to a
    /// top-level definition whose value is not function-typed. Such a capability carries no hidden
    /// effect constraints, so the callee cannot suspend through it. Locals (handler capabilities!),
    /// applications, and constrained impls all fail this.
    fn implicit_is_pure_static(&mut self, expr: ExprId) -> bool {
        let cst::Expr::Variable(path) = &self.context()[expr] else { return false };
        if !matches!(self.context().path_origin(*path), Some(Origin::TopLevelDefinition(_))) {
            return false;
        }
        let typ = self.types.result.maps.expr_types[&expr].follow(&self.types.bindings);
        !matches!(typ, TCType::Function(_))
    }

    /// Release a shared handle the caller owns past the call (`release_T` derived from the argument
    /// expression's TC type). No-op for non-shared types.
    fn emit_shared_release_for_expr(&mut self, value: Value, expr: ExprId) -> Option<Value> {
        let tc_type = self.types.result.maps.expr_types[&expr].follow(&self.types.bindings);
        if let Some((type_id, type_args)) = self.shared_release_target(&tc_type) {
            return Some(self.emit_release_call(value, type_id, &type_args));
        }
        None
    }

    /// A first-class effect operation: projects the operation out of the capability at the head of its own evidence.
    fn effect_op_value_wrapper(&mut self, path_id: PathId, op_index: u32) -> Value {
        let uniform_type = self.convert_path_type(path_id);
        let Type::Function(uniform_ft) = &uniform_type else { return Value::Error };
        let parameters = uniform_ft.parameters.clone();
        let return_type = uniform_ft.return_type.clone();
        let surface_count = parameters.len() - 1;

        let generics_count = self.generics_in_scope.len() as u32;
        let name = Arc::new("effect_op".to_string());
        let id = self.new_isolated_definition(name.clone(), generics_count, uniform_type.clone(), |this| {
            let params = this.push_capability_parameters(&parameters);
            let evidence = params[surface_count];
            let Type::Tuple(fields) = &parameters[surface_count] else {
                unreachable!("effect operation's evidence must contain its capability")
            };
            let index = Instruction::IndexTuple { tuple: evidence, index: 0 };
            let capability = this.push_instruction(index, fields[0].clone());
            let method = this.index_capability_method(capability, op_index);
            let arguments = params[..surface_count].to_vec();
            let instruction = Instruction::CallClosure { closure: method, arguments };
            let result = this.push_instruction(instruction, return_type.clone());
            this.terminate_block(TerminatorInstruction::Return(result));
        });
        self.make_definition_value(id, name, uniform_type)
    }

    /// Wraps a C-shaped extern function value into the uniform evidence convention.
    fn extern_evidence_wrapper(&mut self, target: Value, name: Name, uniform_type: Type) -> Value {
        let Type::Function(uniform_ft) = &uniform_type else { return target };
        let parameters = uniform_ft.parameters.clone();
        let return_type = uniform_ft.return_type.clone();
        let surface_count = parameters.len().saturating_sub(1);
        let Value::Definition(target_id) = target else { return target };

        let generics_count = self.generics_in_scope.len() as u32;
        let name = Arc::new(format!("{name}_wrapper"));
        let id = self.new_isolated_definition(name.clone(), generics_count, uniform_type.clone(), |this| {
            let params = this.push_capability_parameters(&parameters);
            let arguments = params[..surface_count].to_vec();
            let function = Value::Definition(target_id);
            let result = this.push_instruction(Instruction::Call { function, arguments }, return_type.clone());
            this.terminate_block(TerminatorInstruction::Return(result));
        });
        self.make_definition_value(id, name, uniform_type)
    }

    /// Whether an expression is (an annotation of) an `extern` declaration.
    fn rhs_is_extern(&self, expr: ExprId) -> bool {
        match &self.context()[expr] {
            cst::Expr::Extern(_) => true,
            cst::Expr::TypeAnnotation(annotation) => self.rhs_is_extern(annotation.lhs),
            _ => false,
        }
    }

    /// Whether a top-level name is defined as an `extern` symbol.
    fn name_is_extern(&self, name: &TopLevelName) -> bool {
        let (item, context) = GetItem(name.top_level_item).get(self.compiler);
        let cst::TopLevelItemKind::Definition(definition) = &item.kind else { return false };
        let mut expr = definition.rhs;
        loop {
            match &context[expr] {
                cst::Expr::Extern(_) => return true,
                cst::Expr::TypeAnnotation(annotation) => expr = annotation.lhs,
                _ => return false,
            }
        }
    }

    /// The callee's raw effects row.
    fn callee_effects(&self, callee_expr: ExprId) -> TCType {
        // `expr_types` for a `Variable` may hold the post-unification expected type instead.
        let typ = match &self.context()[callee_expr] {
            cst::Expr::Variable(path_id) => self.types.result.maps.path_types[path_id].follow(&self.types.bindings),
            _ => self.types.result.maps.expr_types[&callee_expr].follow(&self.types.bindings),
        };
        match typ {
            TCType::Function(function_type) => function_type.effects.clone(),
            _ => TCType::Effects(Arc::new(Vec::new()), None),
        }
    }

    /// If `value` statically references a definition, returns its id and any instantiation.
    fn as_static_function(&mut self, value: Value) -> Option<(DefinitionId, Option<Arc<Vec<Type>>>)> {
        match value {
            Value::Definition(id) => Some((id, None)),
            Value::InstructionResult(iid) => match &self.current_function().instructions[iid] {
                Instruction::Instantiate(id, bindings) => Some((*id, Some(bindings.clone()))),
                Instruction::Id(Value::Definition(id)) => Some((*id, None)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Inference subtypes function arguments (so the argument's row may be smaller than
    /// the parameter's), but evidence shapes are invariant. When the shapes differ, wrap the
    /// argument in an adapter of the expected shape whose body projects the argument's own
    /// smaller evidence out of the expected evidence.
    fn coerce_argument_evidence(
        &mut self, value: Value, function: &Value, callee_expr: ExprId, parameter_index: usize,
    ) -> Value {
        let Type::Function(callee_ft) = self.type_of_value(function) else { return value };
        let Some(Type::Function(expected_ft)) = callee_ft.parameters.get(parameter_index).cloned() else {
            return value;
        };
        let value_type = self.type_of_value(&value);
        let Type::Function(vt) = &value_type else { return value };
        if vt.parameters == expected_ft.parameters {
            return value;
        }
        // Only a differing trailing evidence parameter is coercible.
        let (Some((value_evidence, value_surface)), Some((expected_evidence, expected_surface))) =
            (vt.parameters.split_last(), expected_ft.parameters.split_last())
        else {
            return value;
        };
        if value_surface != expected_surface || vt.return_type != expected_ft.return_type {
            return value;
        }

        // A closure argument reuses its own environment, so the adapted closure's type matches the expected one exactly.
        let peeked_closure = match value {
            Value::InstructionResult(iid) => match &self.current_function().instructions[iid] {
                Instruction::PackClosure { function, environment } => Some((*function, *environment)),
                _ => None,
            },
            _ => None,
        };
        let static_closure = peeked_closure.and_then(|(function, environment)| {
            self.as_static_function(function).map(|target| (target, function, environment))
        });
        let (target, environment) = match self.as_static_function(value) {
            Some(target) => (target, None),
            None => match static_closure {
                Some((target, function, environment)) => {
                    (target, Some((environment, self.type_of_value(&function), self.type_of_value(&environment))))
                },
                // A dynamic function value: wrap it opaquely and patch the callee's environment-generic instantiation.
                None => {
                    return self.coerce_dynamic_argument_evidence(
                        value,
                        &expected_ft,
                        expected_evidence,
                        value_evidence,
                        callee_expr,
                        function,
                        parameter_index,
                    );
                },
            },
        };
        let (target_id, target_bindings) = target;
        let target_type = match &environment {
            Some((_, function_type, _)) => function_type.clone(),
            None => value_type.clone(),
        };

        let return_type = vt.return_type.clone();
        let value_evidence = value_evidence.clone();
        let expected_evidence = expected_evidence.clone();
        let expected_parameters = expected_ft.parameters.clone();
        let surface_count = expected_surface.len();
        let environment_type = environment.as_ref().map(|(_, _, env_type)| env_type.clone());

        let declared_type = Type::Function(Arc::new(FunctionType {
            parameters: expected_parameters.clone(),
            environment: environment_type.clone().unwrap_or(Type::NO_CLOSURE_ENV),
            return_type: return_type.clone(),
        }));

        let generics_count = self.generics_in_scope.len() as u32;
        let name = Arc::new("evidence_adapter".to_string());
        let id = self.new_isolated_definition(name.clone(), generics_count, declared_type.clone(), |this| {
            let params = this.push_capability_parameters(&expected_parameters);
            let environment_param = environment_type.clone().map(|typ| this.push_capability_parameter(typ));

            let provided = params[surface_count];
            let evidence = this.project_evidence_or_panic(provided, &expected_evidence, &value_evidence);

            let target = match &target_bindings {
                Some(bindings) => {
                    this.push_instruction(Instruction::Instantiate(target_id, bindings.clone()), target_type.clone())
                },
                None => Value::Definition(target_id),
            };
            let mut arguments: Vec<Value> = params[..surface_count].to_vec();
            arguments.push(evidence);
            arguments.extend(environment_param);
            let result = this.push_instruction(Instruction::Call { function: target, arguments }, return_type.clone());
            this.terminate_block(TerminatorInstruction::Return(result));
        });

        let adapted = self.make_definition_value(id, name, declared_type.clone());
        match environment {
            Some((environment, _, _)) => {
                self.push_instruction(Instruction::PackClosure { function: adapted, environment }, declared_type)
            },
            None => adapted,
        }
    }

    /// The adapter closure captures the value as its own environment, so the callee's environment-generic binding must be patched to the adapter's shape.
    #[allow(clippy::too_many_arguments)]
    fn coerce_dynamic_argument_evidence(
        &mut self, value: Value, expected_ft: &FunctionType, expected_evidence: &Type, value_evidence: &Type,
        callee_expr: ExprId, function: &Value, parameter_index: usize,
    ) -> Value {
        let value_type = self.type_of_value(&value);
        let return_type = expected_ft.return_type.clone();
        let expected_parameters = expected_ft.parameters.clone();
        let surface_count = expected_parameters.len() - 1;
        let expected_evidence = expected_evidence.clone();
        let value_evidence = value_evidence.clone();
        let is_closure = value_type.is_closure();

        let declared_type = Type::Function(Arc::new(FunctionType {
            parameters: expected_parameters.clone(),
            environment: value_type.clone(),
            return_type: return_type.clone(),
        }));

        let generics_count = self.generics_in_scope.len() as u32;
        let name = Arc::new("evidence_adapter".to_string());
        let inner_type = value_type.clone();
        let id = self.new_isolated_definition(name.clone(), generics_count, declared_type.clone(), |this| {
            let params = this.push_capability_parameters(&expected_parameters);
            let inner = this.push_capability_parameter(inner_type.clone());

            let provided = params[surface_count];
            let evidence = this.project_evidence_or_panic(provided, &expected_evidence, &value_evidence);

            let mut arguments: Vec<Value> = params[..surface_count].to_vec();
            arguments.push(evidence);
            let instruction = if is_closure {
                Instruction::CallClosure { closure: inner, arguments }
            } else {
                Instruction::Call { function: inner, arguments }
            };
            let result = this.push_instruction(instruction, return_type.clone());
            this.terminate_block(TerminatorInstruction::Return(result));
        });

        let adapter = self.make_definition_value(id, name, declared_type.clone());
        let adapted =
            self.push_instruction(Instruction::PackClosure { function: adapter, environment: value }, declared_type);
        self.patch_environment_binding(function, callee_expr, parameter_index, value_type);
        adapted
    }

    /// Patches a callee's already-built `Instantiate` binding for its `parameter_index`'th parameter's environment generic.
    fn patch_environment_binding(
        &mut self, function: &Value, callee_expr: ExprId, parameter_index: usize, new_env_type: Type,
    ) {
        let Value::InstructionResult(inst_id) = *function else { return };
        let Instruction::Instantiate(callee_id, bindings) = &self.current_function().instructions[inst_id] else {
            return;
        };
        let callee_id = *callee_id;
        let old_bindings = bindings.as_ref().clone();

        let cst::Expr::Variable(path_id) = &self.context()[callee_expr] else { return };
        let Some(Origin::TopLevelDefinition(name)) = self.context().path_origin(*path_id) else { return };
        let checked = TypeCheck(name.top_level_item).get(self.compiler);
        let Some(TCType::Forall(generics, inner)) = checked.result.generalized.get(&name.local_name_id) else {
            return;
        };
        let TCType::Function(callee_fn) = inner.as_ref() else { return };
        let Some(parameter) = callee_fn.parameters.get(parameter_index) else { return };
        let TCType::Function(param_fn) = parameter.typ.follow(&checked.bindings) else { return };
        let TCType::Generic(target) = param_fn.environment.follow(&checked.bindings) else { return };
        let Some(idx) = generics.iter().position(|g| g == target) else { return };
        if idx >= old_bindings.len() {
            return;
        }

        let mut new_bindings = old_bindings;
        new_bindings[idx] = new_env_type;

        let new_typ = self.convert_and_instantiate(inner, generics, &checked.bindings, &new_bindings);

        let current = self.current_function();
        current.instructions[inst_id] = Instruction::Instantiate(callee_id, Arc::new(new_bindings));
        current.instruction_result_types[inst_id] = new_typ;
    }

    /// Builds a needed evidence value out of a larger `provided` one by walking the provided cons chain.
    fn project_evidence(&mut self, provided: Value, provided_type: &Type, needed_type: &Type) -> Option<Value> {
        if provided_type == needed_type {
            return Some(provided);
        }
        // The empty evidence needs nothing from the provided chain.
        if matches!(needed_type, Type::Tuple(fields) if fields.is_empty()) {
            return Some(self.push_instruction(Instruction::MakeTuple(Vec::new()), Type::tuple(Vec::new())));
        }
        let Type::Tuple(needed_fields) = needed_type else { return None };
        let [needed_head, needed_rest] = needed_fields.as_slice() else { return None };

        let mut current = provided;
        let mut current_type = provided_type.clone();
        loop {
            let Type::Tuple(fields) = &current_type else { return None };
            let [head, rest] = fields.as_slice() else { return None };
            let (head, rest) = (head.clone(), rest.clone());
            if head == *needed_head {
                let index = Instruction::IndexTuple { tuple: current, index: 0 };
                let capability = self.push_instruction(index, head.clone());
                let index = Instruction::IndexTuple { tuple: current, index: 1 };
                let rest_value = self.push_instruction(index, rest.clone());
                let rest_evidence = self.project_evidence(rest_value, &rest, needed_rest)?;
                let tuple = Instruction::MakeTuple(vec![capability, rest_evidence]);
                return Some(self.push_instruction(tuple, needed_type.clone()));
            }
            let index = Instruction::IndexTuple { tuple: current, index: 1 };
            current = self.push_instruction(index, rest.clone());
            current_type = rest;
        }
    }

    /// Appends the callee's evidence argument. A callee whose value type has no evidence
    /// slot (externs, `resume`, effect operations) takes none.
    fn append_evidence_argument(&mut self, callee_expr: ExprId, callee_value: &Value, arguments: &mut Vec<Value>) {
        if let Type::Function(function_type) = self.type_of_value(callee_value)
            && function_type.parameters.len() <= arguments.len()
        {
            return;
        }
        let effects = self.callee_effects(callee_expr);
        let evidence = self.build_evidence(&effects);
        arguments.push(evidence);
    }

    /// Builds the evidence tuple for an effect row.
    fn build_evidence(&mut self, effects: &TCType) -> Value {
        let (concretes, end) = self.convert_context().split_row(effects);
        let sources = self.resolve_capability_sources(&concretes);
        let (mut evidence, mut evidence_type) = match end {
            types::RowEnd::Closed => {
                let typ = Type::tuple(Vec::new());
                (self.push_instruction(Instruction::MakeTuple(Vec::new()), typ.clone()), typ)
            },
            types::RowEnd::Generic(_) => {
                let rest = self
                    .capability_bundle
                    .unwrap_or_else(|| panic!("no ambient evidence to forward to a row-polymorphic callee"));
                let typ = self.type_of_value(&rest);
                (rest, typ)
            },
        };
        for source in sources.iter().rev() {
            let capability = self.capability_value(*source);
            let capability_type = self.type_of_value(&capability);
            evidence_type = Type::tuple(vec![capability_type, evidence_type]);
            evidence = self.push_instruction(Instruction::MakeTuple(vec![capability, evidence]), evidence_type.clone());
        }
        evidence
    }

    /// Resolves each of `required`'s concrete effects to a [CapabilitySource] against this
    /// function's effects and any captured handles.
    fn resolve_capability_sources(&self, required: &[TCType]) -> Vec<CapabilitySource> {
        mapvec(required, |effect| self.resolve_one_capability_source(effect))
    }

    fn resolve_one_capability_source(&self, effect: &TCType) -> CapabilitySource {
        let key = self.capability_key(effect);
        if let Some(i) = self.effects.iter().position(|(owned, _)| self.capability_key(owned) == key) {
            return CapabilitySource::CanClause(i);
        }
        for (handle_expr, (owned, _)) in &self.handle_capabilities {
            if self.capability_key(owned) == key {
                return CapabilitySource::Handle(*handle_expr);
            }
        }
        panic!("no capability for effect {key:?} in scope at this call site")
    }

    /// Canonicalizes a concrete effect type to a stable comparison key.
    fn capability_key(&self, effect_ty: &TCType) -> TCType {
        let followed = effect_ty.follow_all(&self.types.bindings);
        let followed = match &followed {
            TCType::Effects(list, None) if list.len() == 1 => list[0].clone(),
            _ => followed,
        };
        Self::widen_reference_kinds(&followed)
    }

    /// Widens every reference-kind occurrence in `ty` to `Ref` (mut/uniq/imm/ref all share one
    /// representation) so e.g. `Emit (mut a)` and `Emit (ref a)` key to the same capability slot.
    fn widen_reference_kinds(ty: &TCType) -> TCType {
        match ty {
            TCType::Primitive(type_inference::types::PrimitiveType::Reference(_)) => TCType::REF,
            TCType::Application(constructor, args) => TCType::Application(
                Arc::new(Self::widen_reference_kinds(constructor)),
                Arc::new(mapvec(args.iter(), Self::widen_reference_kinds)),
            ),
            TCType::Tuple(elements) => TCType::Tuple(Arc::new(mapvec(elements.iter(), Self::widen_reference_kinds))),
            other => other.clone(),
        }
    }

    fn capability_value(&self, source: CapabilitySource) -> Value {
        match source {
            CapabilitySource::CanClause(i) => self.effects[i].1,
            CapabilitySource::Handle(id) => {
                self.handle_capabilities.get(&id).unwrap_or_else(|| panic!("no capability for handle {id:?}")).1
            },
        }
    }

    /// The concrete effects a handler body/branch needs beyond what it handles.
    /// A body excludes its own handled effect(s) while a branch excludes nothing.
    fn suppressed_lambda_needed_capabilities(
        &self, tc_effects: &TCType, handle_body_handler_name: Option<NameId>,
    ) -> Vec<TCType> {
        let (effects, _) = self.convert_context().split_row(tc_effects);

        let excluded_keys: Vec<TCType> = match handle_body_handler_name {
            Some(handler_name) => {
                let h_tc_type = self.types.result.maps.name_types[&handler_name].follow_all(&self.types.bindings);
                match &h_tc_type {
                    TCType::Effects(h_list, _) => h_list.iter().map(|t| self.capability_key(t)).collect(),
                    other => vec![self.capability_key(other)],
                }
            },
            None => Vec::new(),
        };

        effects.into_iter().filter(|effect| !excluded_keys.contains(&self.capability_key(effect))).collect()
    }

    /// Extends a lambda's environment type with trailing fields.
    fn extend_environment_with_fields(&self, full_type: Type, extra_fields: impl IntoIterator<Item = Type>) -> Type {
        let Type::Function(ft) = &full_type else { unreachable!("Lambda does not have a function type") };
        let mut env_fields: Vec<Type> = match &ft.environment {
            Type::Tuple(fields) => (**fields).clone(),
            _ => Vec::new(),
        };
        env_fields.extend(extra_fields);
        Type::Function(Arc::new(FunctionType {
            parameters: ft.parameters.clone(),
            environment: Type::tuple(env_fields),
            return_type: ft.return_type.clone(),
        }))
    }

    /// Extends a lambda's environment type with a trailing field per needed capability value.
    fn extend_environment_with_capabilities(&self, full_type: Type, needed_capability_values: &[Value]) -> Type {
        if needed_capability_values.is_empty() {
            return full_type;
        }
        self.extend_environment_with_fields(full_type, needed_capability_values.iter().map(|v| self.type_of_value(v)))
    }

    /// Extends a lambda's environment type with one trailing field for the captured enclosing evidence rest.
    fn extend_environment_with_bundle(&self, full_type: Type, bundle_type: Type) -> Type {
        self.extend_environment_with_fields(full_type, std::iter::once(bundle_type))
    }

    /// Extracts the `op_index`'th method out of a capability/dictionary tuple value.
    fn index_capability_method(&mut self, cap_value: Value, op_index: u32) -> Value {
        let cap_type = self.type_of_value(&cap_value);
        let method_type = match &cap_type {
            Type::Tuple(fields) => fields.get(op_index as usize).cloned().unwrap_or_else(|| {
                panic!("ability method call: cap tuple has no slot {op_index} (cap_type = {cap_type})")
            }),
            // Single-method abilities can collapse to a bare function when type inference
            // strips the surrounding tuple. Fall back to the call's expected result-type wiring.
            _ => cap_type.clone(),
        };
        self.push_instruction(Instruction::IndexTuple { tuple: cap_value, index: op_index }, method_type)
    }

    /// Emits the `IndexTuple cap op_index + CallClosure` sequence for a trait or effect method call.
    fn emit_indexed_method_call(
        &mut self, callee_expr: Option<ExprId>, cap_value: Value, op_index: u32, mut arguments: Vec<Value>,
        result_type: Type, diverges: bool,
    ) -> Value {
        let method = self.index_capability_method(cap_value, op_index);
        let method_type = self.type_of_value(&method);
        if let Some(callee_expr) = callee_expr {
            self.append_evidence_argument(callee_expr, &method, &mut arguments);
        }

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

    /// `arguments` must contain the operation args followed by the implicit dictionary value
    fn emit_trait_method_call(
        &mut self, callee_expr: ExprId, effect_op: DefinitionId, op_index: u32, mut arguments: Vec<Value>,
        result_type: Type, diverges: bool,
    ) -> Value {
        self.effect_op_indices.insert(effect_op, op_index);
        let cap_value = arguments.pop().expect("trait method call: no implicit cap argument");
        self.emit_indexed_method_call(Some(callee_expr), cap_value, op_index, arguments, result_type, diverges)
    }

    fn emit_effect_op_call(
        &mut self, callee_expr: ExprId, effect_op: DefinitionId, op_index: u32, arguments: Vec<Value>,
        result_type: Type, diverges: bool,
    ) -> Value {
        self.effect_op_indices.insert(effect_op, op_index);
        let effect_type = self.effect_type_of_op(callee_expr);
        let source = self.resolve_one_capability_source(&effect_type);
        let cap_value = self.capability_value(source);
        self.emit_indexed_method_call(None, cap_value, op_index, arguments, result_type, diverges)
    }

    /// The concrete effect an operation reference performs, read off its own singleton effects row.
    fn effect_type_of_op(&self, callee_expr: ExprId) -> TCType {
        let typ = self.types.result.maps.expr_types[&callee_expr].follow(&self.types.bindings);
        let TCType::Function(function_type) = typ else {
            panic!("effect_type_of_op: operation is not a function type");
        };
        let TCType::Effects(list, _) = function_type.effects.follow(&self.types.bindings) else {
            panic!("effect_type_of_op: operation has no effects row");
        };
        list.first().unwrap_or_else(|| panic!("effect_type_of_op: operation has an empty effects row")).clone()
    }

    /// Resolves a concrete effect `Type` to its capability tuple type.
    fn effect_capability_tuple_type_of(&self, effect_type: &TCType) -> Type {
        self.convert_context().effect_capability_tuple_type_of(effect_type)
    }

    /// True if `item` is the Prelude's `Drop` trait definition.
    fn is_prelude_drop_trait(&mut self, item: TopLevelId) -> bool {
        if item.source_file != crate::name_resolution::namespace::SourceFileId::prelude() {
            return false;
        }
        let (raw_item, context) = GetItemRaw(item).get(self.compiler);
        match &raw_item.kind {
            cst::TopLevelItemKind::TraitDefinition(definition) => context.names[definition.name].as_ref() == "Drop",
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
        let mut arguments = mapvec(call.arguments.iter().enumerate(), |(i, argument)| {
            let value = self.lower_drop_capability(argument.expr, env_frees);
            self.coerce_argument_evidence(value, &function, call.function, i)
        });
        // A constrained impl is an ordinary Ante function, so its converted type carries the
        // uniform trailing evidence parameter. Hand-lowering the application has to supply it
        // just as `call` does, or the callee is invoked one argument short.
        self.append_evidence_argument(call.function, &function, &mut arguments);

        // Coercing argument evidence above may have patched `function`'s own type in place
        // (see `patch_environment_binding`), so read the result type off the callee.
        let function_type = self.type_of_value(&function);
        let result_type = match &function_type {
            Type::Function(function_type) => function_type.return_type.clone(),
            _ => self.expr_type(expr),
        };

        let instruction = if function_type.is_closure() {
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

    /// Whether a top-level item is a trait, effect, or neither
    fn ability_kind(&mut self, item: TopLevelId) -> AbilityKind {
        if let Some(kind) = self.ability_defs.get(&item).copied() {
            return kind;
        }
        let (cst_item, _) = GetItemRaw(item).get(self.compiler);
        let kind = match &cst_item.kind {
            cst::TopLevelItemKind::TraitDefinition(_) => AbilityKind::Trait,
            cst::TopLevelItemKind::EffectDefinition(_) => AbilityKind::Effect,
            _ => AbilityKind::NotAbility,
        };
        self.ability_defs.insert(item, kind);
        kind
    }

    /// If `path` resolves to a trait or effect operation declaration, return its
    /// [DefinitionId], its position within its ability's body (for indexing its capability tuple),
    /// and the ability's kind. Otherwise return None.
    fn try_resolve_ability_method(&mut self, path: PathId) -> Option<(DefinitionId, u32, AbilityKind)> {
        let origin = self.context().path_origin(path)?;
        let Origin::TopLevelDefinition(name) = origin else { return None };

        let kind = self.ability_kind(name.top_level_item);
        if kind == AbilityKind::NotAbility {
            return None;
        }

        let (item, _) = GetItemRaw(name.top_level_item).get(self.compiler);
        if let cst::TopLevelItemKind::TraitDefinition(effect) | cst::TopLevelItemKind::EffectDefinition(effect) =
            &item.kind
            && let Some(op_index) = effect.body.iter().position(|d| d.name == name.local_name_id)
        {
            let id = self.get_definition_id(&name);
            return Some((id, op_index as u32, kind));
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
                for argument in arguments {
                    self.collect_pattern_names(*argument, out);
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

    /// Builds a new isolated top-level definition, saving and restoring this builder's local
    /// scope state (locals, mutability, capabilities) around it.
    fn new_isolated_definition(
        &mut self, name: Name, generic_count: u32, typ: Type, f: impl FnOnce(&mut Self),
    ) -> DefinitionId {
        let saved = self.take_scope();
        let id = self.new_definition(name, None, generic_count, typ, f);
        self.restore_scope(saved);
        id
    }

    /// Takes this builder's local scope state, leaving it empty. Paired with [Self::restore_scope].
    fn take_scope(&mut self) -> SavedScope {
        SavedScope {
            local_variables: std::mem::take(&mut self.local_variables),
            mutable_locals: std::mem::take(&mut self.mutable_locals),
            own_row: std::mem::take(&mut self.effects),
            handle_capabilities: std::mem::take(&mut self.handle_capabilities),
            capability_bundle: self.capability_bundle.take(),
        }
    }

    /// Restores local scope state previously taken by [Self::take_scope].
    fn restore_scope(&mut self, saved: SavedScope) {
        self.local_variables = saved.local_variables;
        self.mutable_locals = saved.mutable_locals;
        self.effects = saved.own_row;
        self.handle_capabilities = saved.handle_capabilities;
        self.capability_bundle = saved.capability_bundle;
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
        self.lambda_impl(lambda, name_id, name, expr, is_global, None, false)
    }

    /// When `handle_body_handler_name` is `Some(h)`, this lambda is the body of a `handle` expression.
    /// Inject a prelude that loads the capability from the coroutine's user_data and binds it
    /// to `h` in `local_variables` so the body's references to `h` resolve.
    ///
    /// `is_handler_branch` and `handle_body_handler_name` both suppress capability parameters.
    fn lambda_impl(
        &mut self, lambda: &cst::Lambda, name_id: Option<NameId>, name: Option<Name>, expr: ExprId, is_global: bool,
        handle_body_handler_name: Option<NameId>, is_handler_branch: bool,
    ) -> Value {
        let name = name.unwrap_or_else(|| Arc::new("lambda".to_string()));
        let suppress_capabilities = handle_body_handler_name.is_some() || is_handler_branch;
        let tc_function_type = match self.types.result.maps.expr_types[&expr].follow(&self.types.bindings) {
            TCType::Function(tc_function_type) => tc_function_type.clone(),
            _ => unreachable!("Lambda does not have a function type"),
        };
        let tc_effects = tc_function_type.effects.clone();
        let (own_concretes, own_end) = self.convert_context().split_row(&tc_effects);

        // Resolved against the enclosing scope's still-current tables, before they're taken below.
        let needed_capabilities = if suppress_capabilities {
            let effects = self.suppressed_lambda_needed_capabilities(&tc_effects, handle_body_handler_name);
            mapvec(effects, |effect| {
                let value = self.capability_value(self.resolve_one_capability_source(&effect));
                (effect, value)
            })
        } else {
            Vec::new()
        };

        let needed_capability_values = mapvec(&needed_capabilities, |(_, v)| *v);

        // A suppressed lambda still needing the ambient evidence rest must capture it through the environment.
        let needs_bundle_capture =
            suppress_capabilities && matches!(own_end, types::RowEnd::Generic(_)) && self.capability_bundle.is_some();
        let bundle_capture_value = if needs_bundle_capture { self.capability_bundle } else { None };
        let bundle_capture_type = bundle_capture_value.map(|v| self.type_of_value(&v));

        // A handler branch's trailing `resume` parameter is a coroutine primitive: C-shaped, no evidence.
        let parameter_types: Vec<Type> = mapvec(lambda.parameters.iter().enumerate(), |(i, parameter)| {
            let parameter_type = &self.types.result.maps.pattern_types[&parameter.pattern];
            if is_handler_branch && i + 1 == lambda.parameters.len() {
                self.convert_context().convert_c_function_type(parameter_type)
            } else {
                self.convert_type(parameter_type, None)
            }
        });
        let evidence_type = self.convert_context().evidence_type(&tc_effects);

        // A suppressed lambda routes capabilities through the environment instead of an evidence parameter.
        let full_type = {
            let mut parameters = parameter_types.clone();
            if !suppress_capabilities {
                parameters.push(evidence_type.clone());
            }
            self.convert_context().build_function_type(&tc_function_type, parameters)
        };
        let full_type = self.extend_environment_with_capabilities(full_type, &needed_capability_values);
        let full_type = match &bundle_capture_type {
            Some(bundle_type) => self.extend_environment_with_bundle(full_type, bundle_type.clone()),
            None => full_type,
        };
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
        let old_loop_targets = std::mem::take(&mut self.loop_targets);
        let saved_scope = self.take_scope();

        let mut captured_values = needed_capability_values.clone();
        captured_values.extend(bundle_capture_value);

        let id = self.new_definition(name.clone(), name_id, generics_count, full_type.clone(), |this| {
            for (i, parameter) in lambda.parameters.iter().enumerate() {
                this.push_parameter(parameter_types[i].clone());

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

            // Push the evidence parameter: a tuple of required capabilities
            if !suppress_capabilities {
                let mut current = this.push_capability_parameter(evidence_type.clone());
                let mut current_type = evidence_type.clone();
                for effect in &own_concretes {
                    let Type::Tuple(fields) = &current_type else {
                        unreachable!("evidence type shorter than its row's effects")
                    };
                    let (capability_type, rest_type) = (fields[0].clone(), fields[1].clone());
                    let index = Instruction::IndexTuple { tuple: current, index: 0 };
                    let capability = this.push_instruction(index, capability_type);
                    this.effects.push((effect.clone(), capability));
                    let index = Instruction::IndexTuple { tuple: current, index: 1 };
                    current = this.push_instruction(index, rest_type.clone());
                    current_type = rest_type;
                }
                if matches!(own_end, types::RowEnd::Generic(_)) {
                    this.capability_bundle = Some(current);
                }
            }

            let env_is_pointer =
                matches!(function_type.environment, Type::Primitive(crate::mir::PrimitiveType::Pointer));
            let free_vars = this.context().get_closure_environment(expr);
            let has_captures = free_vars.is_some() || !needed_capabilities.is_empty() || bundle_capture_type.is_some();
            let needs_env_param = has_captures || env_is_pointer;
            let pushed_capability_count = if suppress_capabilities { 0 } else { 1 };
            let env_param_index = lambda.parameters.len() as u32 + pushed_capability_count;
            if needs_env_param {
                if let Some(env) = function_type.environment() {
                    this.push_parameter(env.clone());
                }

                if has_captures {
                    let empty_free_vars = BTreeSet::new();
                    let free_vars = free_vars.unwrap_or(&empty_free_vars);
                    let environment = Value::Parameter(this.current_block, env_param_index);
                    this.unpack_closure_environment(
                        free_vars.iter().copied(),
                        &needed_capabilities,
                        bundle_capture_type.clone(),
                        environment,
                    );

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
                let h_tc_type = this.types.result.maps.name_types[&handler_name].follow_all(&this.types.bindings);
                let h_type = this.effect_capability_tuple_type_of(&h_tc_type);
                let cap = this.push_instruction(Instruction::Capability, h_type);
                this.local_variables.insert(handler_name, cap);
                this.handle_capabilities.insert(expr, (h_tc_type, cap));
            }

            // Pre-populate the self-reference so recursive calls within the body
            // (e.g. `recur` in desugared loop expressions) can resolve via Origin::Local.
            // The entry is discarded automatically when the saved scope is restored after
            // `new_definition` returns.
            if let Some(self_name_id) = name_id {
                let self_def_id = this.current_function.as_ref().unwrap().id;
                let mut self_value = Value::Definition(self_def_id);
                if needs_env_param {
                    let environment = Value::Parameter(this.current_block, env_param_index);
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

        self.loop_targets = old_loop_targets;
        self.restore_scope(saved_scope);

        // Generic lambdas inherit generics from the surrounding context; instantiate manually.
        let mut value = self.make_definition_value(id, name, full_type.clone());
        if !is_global && !self.generics_in_scope.is_empty() {
            let bindings = Arc::new(mapvec(0..self.generics_in_scope.len() as u32, |i| Type::Generic(Generic(i))));
            value = self.push_instruction(Instruction::Instantiate(id, bindings), full_type.clone());
        }
        let free_vars = self.context().get_closure_environment(expr).cloned();
        let env_type = function_type.environment.clone();
        let env_is_pointer = matches!(env_type, Type::Primitive(crate::mir::PrimitiveType::Pointer));
        let has_captures = free_vars.is_some() || !captured_values.is_empty();
        if has_captures || env_is_pointer {
            let environment = if has_captures {
                let free_vars = free_vars.unwrap_or_default();
                self.pack_closure_environment(&free_vars, &captured_values, is_move, &env_type)
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

    /// Packs each given variable, plus any needed capability values, into a closure environment.
    /// When `env_type` is a pointer, the capture tuple is heap-allocated (via [Instruction::AllocShared])
    /// and the returned value is the resulting pointer. Otherwise returns the tuple directly.
    fn pack_closure_environment(
        &mut self, free_vars: &BTreeSet<NameId>, capability_values: &[Value], is_move: bool, env_type: &Type,
    ) -> Value {
        assert!(!free_vars.is_empty() || !capability_values.is_empty());

        let mut values = mapvec(free_vars, |var| {
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
        values.extend_from_slice(capability_values);
        let types = mapvec(&values, |value| self.type_of_value(value));
        let tuple = self.push_instruction(Instruction::MakeTuple(values), Type::tuple(types));

        if matches!(env_type, Type::Primitive(crate::mir::PrimitiveType::Pointer)) {
            self.push_instruction(Instruction::AllocShared(tuple), Type::POINTER)
        } else {
            tuple
        }
    }

    /// Unpack a closure environment parameter, binding each captured name and needed capability
    /// to its value. Each unpacked capability is pushed onto `self.effects` in order.
    fn unpack_closure_environment(
        &mut self, free_vars: impl ExactSizeIterator<Item = NameId> + Clone, capabilities: &[(TCType, Value)],
        bundle_type: Option<Type>, environment: Value,
    ) {
        let free_vars_len = free_vars.len();
        let env_value =
            if matches!(self.type_of_value(&environment), Type::Primitive(crate::mir::PrimitiveType::Pointer)) {
                let mut field_types: Vec<Type> = free_vars
                    .clone()
                    .map(|var| {
                        let tc_type = &self.types.result.maps.name_types[&var];
                        self.convert_type(tc_type, None)
                    })
                    .collect();
                field_types.extend(capabilities.iter().map(|(_, v)| self.type_of_value(v)));
                field_types.extend(bundle_type.clone());
                let tuple_type = Type::tuple(field_types);
                self.push_instruction(Instruction::Deref(environment), tuple_type)
            } else {
                environment
            };

        let Type::Tuple(env_fields) = self.type_of_value(&env_value) else { unreachable!() };
        let expected_len = free_vars_len + capabilities.len() + bundle_type.is_some() as usize;
        assert_eq!(env_fields.len(), expected_len);

        for (i, (var, env_field)) in free_vars.zip(env_fields.iter().cloned()).enumerate() {
            let index = Instruction::IndexTuple { tuple: env_value, index: i as u32 };
            let result = self.push_instruction(index, env_field);
            let existing = self.local_variables.insert(var, result);
            assert!(existing.is_none(), "Closure is overwriting values from the outer scope");
        }
        for (i, (effect, _)) in capabilities.iter().enumerate() {
            let field_ty = env_fields[free_vars_len + i].clone();
            let index = Instruction::IndexTuple { tuple: env_value, index: (free_vars_len + i) as u32 };
            let result = self.push_instruction(index, field_ty);
            self.effects.push((effect.clone(), result));
        }
        if bundle_type.is_some() {
            let idx = free_vars_len + capabilities.len();
            let field_ty = env_fields[idx].clone();
            let index = Instruction::IndexTuple { tuple: env_value, index: idx as u32 };
            let result = self.push_instruction(index, field_ty);
            self.capability_bundle = Some(result);
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
                self.lambda_impl(&body_lambda, None, None, handle.expression, false, Some(handle.handler_name), false)
            },
            _ => unreachable!("handle.expression should be a Lambda after parsing"),
        };

        let cases = mapvec(&handle.cases, |(pattern, branch)| {
            let (effect_op, op_index, _) =
                self.try_resolve_ability_method(pattern.function).expect("Couldn't find effect op in MIR handle");
            self.effect_op_indices.insert(effect_op, op_index);
            let handler = match &self.context()[*branch] {
                cst::Expr::Lambda(branch_lambda) => {
                    let branch_lambda = branch_lambda.clone();
                    self.lambda_impl(&branch_lambda, None, None, *branch, false, None, true)
                },
                _ => unreachable!("handler branch should be a Lambda after parsing"),
            };
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
            let mut arguments = vec![current, rhs];
            self.append_evidence_argument(op_expr, &function, &mut arguments);
            let instruction = if self.type_of_value(&function).is_closure() {
                Instruction::CallClosure { closure: function, arguments }
            } else {
                Instruction::Call { function, arguments }
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
        let typ = self.convert_context().convert_c_function_type(&self.types.result.maps.expr_types[&id]);
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
        if type_definition.kind.is_effect() {
            return;
        }

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
            self.set_generics_in_scope(&constructor_type);

            let parameters = match constructor_type.ignore_forall() {
                TCType::Function(function) => function.parameters.as_slice(),
                _ => &[],
            };

            let shared = type_definition.shared;
            self.define_type_constructor(constructor_name, &constructor_type, parameters, tag, shared);
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
        if type_definition.kind.is_ability() {
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
        // parameters with these -- passing `None` would instantiate them as fresh, unbound type
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
        let fn_type = Self::release_function_mir_type();

        let old_scope = std::mem::take(&mut self.local_variables);
        let old_mutables = std::mem::take(&mut self.mutable_locals);

        // A self-recursive shared type (`Cons I32 L`) must not tear down with one native frame per
        // node -- a long list overflows the stack. When any field releases through this same type
        // instantiation, lower the release as a pointer-chasing loop over the LAST such field per
        // variant (non-last self fields keep their recursive calls, so trees consume depth, not
        // size).
        let has_self_tail = self.type_has_self_release_field(&generic_args);

        let id = self.new_definition(name, Some(NameId::RELEASE_FUNCTION), generic_count, fn_type, |this| {
            this.push_parameter(Type::POINTER);
            let handle = Value::Parameter(this.current_block, 0);

            if has_self_tail {
                // cur/next live in stack slots (the same shape `while_` lowers to):
                //   header:  cur = *cur_slot; if !RcDecrement(cur) -> exit
                //   glue:    *next_slot = null; <release fields, the tail deferred into
                //            next_slot>; FreeShared(cur); if *next_slot == null -> exit
                //   advance: *cur_slot = next; jmp header
                // Immortal statics (`Nil`) end the chase at RcDecrement (a count-0 sentinel
                // is left alone); the null check covers variants with no self field.
                let null =
                    this.push_instruction(Instruction::Transmute(Value::Integer(IntConstant::Usz(0))), Type::POINTER);
                let next_slot = this.push_instruction(Instruction::StackAlloc(null), Type::POINTER);
                let cur_slot = this.push_instruction(Instruction::StackAlloc(handle), Type::POINTER);
                let header = this.push_block_no_params();
                let glue_block = this.push_block_no_params();
                let advance = this.push_block_no_params();
                let exit = this.push_block_no_params();
                this.terminate_block(TerminatorInstruction::jmp_no_args(header));

                this.switch_to_block(header);
                let cur = this.push_instruction(Instruction::Deref(cur_slot), Type::POINTER);
                let reached_zero = this.push_instruction(Instruction::RcDecrement(cur), Type::BOOL);
                this.terminate_block(TerminatorInstruction::if_(reached_zero, glue_block, exit, exit));

                this.switch_to_block(glue_block);
                this.push_instruction(Instruction::Store { pointer: next_slot, value: null }, Type::UNIT);
                this.build_release_glue(cur, &self_tc, &generic_args, Some(next_slot));
                this.push_instruction(Instruction::FreeShared(cur), Type::UNIT);
                let next = this.push_instruction(Instruction::Deref(next_slot), Type::POINTER);
                let next_int = this.push_instruction(Instruction::Transmute(next), Type::int(IntegerKind::Usz));
                let is_null = this
                    .push_instruction(Instruction::EqInt(next_int, Value::Integer(IntConstant::Usz(0))), Type::BOOL);
                this.terminate_block(TerminatorInstruction::if_(is_null, exit, advance, exit));

                this.switch_to_block(advance);
                this.push_instruction(Instruction::Store { pointer: cur_slot, value: next }, Type::UNIT);
                this.terminate_block(TerminatorInstruction::jmp_no_args(header));

                this.switch_to_block(exit);
                this.terminate_block(TerminatorInstruction::Return(Value::Unit));
            } else {
                let reached_zero = this.push_instruction(Instruction::RcDecrement(handle), Type::BOOL);
                let glue_block = this.push_block_no_params();
                let cont = this.push_block_no_params();
                this.terminate_block(TerminatorInstruction::if_(reached_zero, glue_block, cont, cont));

                // Last reference: release the pointee's shared fields, then free the whole block.
                this.switch_to_block(glue_block);
                this.build_release_glue(handle, &self_tc, &generic_args, None);
                this.push_instruction(Instruction::FreeShared(handle), Type::UNIT);
                this.terminate_block(TerminatorInstruction::jmp_no_args(cont));

                this.switch_to_block(cont);
                this.terminate_block(TerminatorInstruction::Return(Value::Unit));
            }
        });

        self.local_variables = old_scope;
        self.mutable_locals = old_mutables;
        self.name_to_id.insert(TopLevelName::new(self.top_level_id, NameId::RELEASE_FUNCTION), id);
    }

    /// True when a directly-releasable field of this type's body is this same type applied to
    /// the same generics -- the self-recursive shape whose release must be a loop.
    fn type_has_self_release_field(&self, generic_args: &[TCType]) -> bool {
        let args = (!generic_args.is_empty()).then_some(generic_args);
        let is_self = |this: &Self, tc: &TCType| {
            this.shared_release_target(tc)
                .is_some_and(|(id, args)| id == this.top_level_id && args.as_slice() == generic_args)
        };
        match self.top_level_id.type_body(args, self.compiler) {
            crate::type_inference::TypeBody::Product { fields, .. } => {
                fields.iter().any(|(_, tc)| is_self(self, tc))
            },
            crate::type_inference::TypeBody::Sum(variants) => {
                variants.iter().any(|(_, payloads)| payloads.iter().any(|tc| is_self(self, tc)))
            },
        }
    }

    /// Emit the pointee-field releases for a shared type's `release_T`. Derefs the handle to the
    /// pointee layout, then walks it via [Self::release_inline_body]. Leaves the builder positioned
    /// at a single continuation block. `tail_slot`, when set, receives the active variant's last
    /// self-recursive handle via a Store instead of a recursive release call.
    fn build_release_glue(&mut self, handle: Value, self_tc: &TCType, generic_args: &[TCType], tail_slot: Option<Value>) {
        let inner = self.deref_if_shared(handle, self_tc);
        // The pointee is processed as an inline aggregate of this type -- but not through
        // [Self::release_inline_value], which would re-enter `release_T` recursively (a self-call).
        self.release_inline_body(inner, self.top_level_id, generic_args, tail_slot);
    }

    /// Release every shared handle reachable from `value`, an inline aggregate value laid out as the
    /// body of user type `type_id` applied to `args`. For a product this releases each field in
    /// place; for a sum it switches on the tag and releases the active variant's payloads, leaving
    /// the builder at a single merge block. Directly-shared fields become recursive `release_T`
    /// calls; nested non-shared aggregates (`Maybe (shared T)`, tuples, …) are traversed inline so
    /// the shared values they wrap are still released. Bare generics are skipped (leaked).
    fn release_inline_body(&mut self, value: Value, type_id: TopLevelId, args: &[TCType], tail_slot: Option<Value>) {
        // With a tail slot active, the LAST field of each variant that releases through this same
        // type instantiation is stored into the slot instead of released (the release loop
        // pointer-chases it). Earlier self fields keep recursive calls.
        let is_self_field = |this: &Self, tc: &TCType| {
            tail_slot.is_some()
                && this
                    .shared_release_target(tc)
                    .is_some_and(|(id, targs)| id == this.top_level_id && targs.as_slice() == args)
        };
        // Substitute the type's parameters with these args; `None` would instantiate them as fresh
        // unbound type variables (→ `Type::Error` at codegen).
        let args_opt = (!args.is_empty()).then_some(args);
        match type_id.type_body(args_opt, self.compiler) {
            crate::type_inference::TypeBody::Product { fields, .. } => {
                let tail_index = fields.iter().rposition(|(_, tc)| is_self_field(self, tc));
                let variant = self.extract_variant(value, 0);
                for (i, (_, field_tc)) in fields.iter().enumerate() {
                    if Some(i) == tail_index {
                        self.defer_release_into_slot(variant, i as u32, field_tc, tail_slot.unwrap());
                    } else {
                        self.release_inline_field(variant, i as u32, field_tc);
                    }
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
                        let tail_index = payloads.iter().rposition(|tc| is_self_field(self, tc));
                        let variant = self.extract_variant(value, idx);
                        for (j, payload_tc) in payloads.iter().enumerate() {
                            if Some(j) == tail_index {
                                self.defer_release_into_slot(variant, j as u32, payload_tc, tail_slot.unwrap());
                            } else {
                                self.release_inline_field(variant, j as u32, payload_tc);
                            }
                        }
                    }
                    self.terminate_block(TerminatorInstruction::jmp_no_args(end));
                }
                self.switch_to_block(end);
            },
        }
    }

    /// Project field `index` out of `tuple` and store its handle into the release
    /// loop's tail slot (deferring it to the pointer chase) instead of releasing it.
    fn defer_release_into_slot(&mut self, tuple: Value, index: u32, field_tc: &TCType, slot: Value) {
        let field_type = self.convert_type(field_tc, None);
        let field = self.push_instruction(Instruction::IndexTuple { tuple, index }, field_type);
        self.push_instruction(Instruction::Store { pointer: slot, value: field }, Type::UNIT);
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
            // Nested aggregates never defer: only the release loop's own top-level pointee
            // walk pointer-chases.
            self.release_inline_body(value, type_id, &type_args, None);
        }
    }

    /// The MIR signature every synthesized `release_T` has: the handle is a shared pointer whatever
    /// the element type is, so `fn(Pointer) -> Unit` serves them all and genericity lives in the
    /// definition's generic count plus its body's `Instantiate`s.
    ///
    /// C-shaped on purpose -- no trailing evidence parameter. A release performs no effects, and
    /// every call to one is emitted by the compiler, so there is no first-class use that would need
    /// the uniform shape. The definition, [Self::emit_release_call] and the `release_T` case in
    /// [Self::variable] must agree exactly, so all three read the signature from here: reading it
    /// off `path_types` instead would hand a reference the evidence parameter the definition lacks.
    fn release_function_mir_type() -> Type {
        Type::Function(Arc::new(crate::mir::FunctionType {
            parameters: vec![Type::POINTER],
            environment: Type::NO_CLOSURE_ENV,
            return_type: Type::UNIT,
        }))
    }

    /// Emit `release_T(handle)` for shared type `type_id` applied to `type_args` (instantiating the
    /// generic release function when the type is generic). Returns the call so a borrowing call
    /// site can record it as the release half of an elidable pair.
    fn emit_release_call(&mut self, handle: Value, type_id: TopLevelId, type_args: &[TCType]) -> Value {
        let release_name = TopLevelName::new(type_id, NameId::RELEASE_FUNCTION);
        let release_id = self.get_definition_id(&release_name);
        let fn_type = Self::release_function_mir_type();
        let callee = if type_args.is_empty() {
            self.make_definition_value(release_id, Arc::new("release".to_string()), fn_type)
        } else {
            let mir_args = mapvec(type_args, |a| self.convert_type(a, None));
            self.push_instruction(Instruction::Instantiate(release_id, Arc::new(mir_args)), fn_type)
        };
        self.push_instruction(Instruction::Call { function: callee, arguments: vec![handle] }, Type::UNIT)
    }

    /// If `typ` resolves to a non-shared user-defined type (a product/sum whose inline layout may
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
        // The (ignored) evidence parameter every function takes.
        if !field_types.is_empty() {
            self.push_parameter(Type::tuple(Vec::new()));
        }

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
            self.set_generics_in_scope(&constructor_type);
            let generic_count = self.generics_in_scope.len() as u32;
            let constructor_mir_type = self.convert_type(&constructor_type, None);

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

                // Callers see the method's TC type: surface params, implicit receiver, then
                // the uniform evidence parameter last.
                let mut wrapper_params = value_param_types;
                let evidence = wrapper_params.pop().unwrap_or_else(|| Type::tuple(Vec::new()));
                wrapper_params.push(struct_type.clone());
                wrapper_params.push(evidence);
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
                    // Second-to-last parameter is the implicit struct receiver (evidence is last).
                    let struct_index = n_params - 2;
                    let struct_param = Value::Parameter(BlockId::ENTRY_BLOCK, struct_index as u32);
                    let extracted = this.push_instruction(
                        Instruction::IndexTuple { tuple: struct_param, index: i as u32 },
                        field_type_clone,
                    );
                    let value_args = mapvec((0..n_params).filter(|j| *j != struct_index), |j| {
                        Value::Parameter(BlockId::ENTRY_BLOCK, j as u32)
                    });
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
