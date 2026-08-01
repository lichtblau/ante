//! Post-monomorphization elision of borrow pairs.
//!
//! Call sites emit a retain + post-call-release pair around a borrowing-parameter argument whenever
//! the call's implicit capabilities are not provably pure pre-mono -- in particular inside generic
//! functions whose capabilities are their own parameters (`ins`'s recursion pays a pair per hop
//! because `{Cmp t}` is a local there). After monomorphization the whole program is at hand and the
//! capability values are concrete, so this pass re-judges each recorded pair
//! ([`crate::mir::BorrowPair`], carried inside each [`Definition`] through every cloning path) and
//! removes the retain and release when all of the call's implicit arguments are pure:
//!
//! - an argument that resolves (through `Instruction::Id` chains, monomorphization's
//!   residue of `Instantiate`) to a top-level definition is pure -- static impls cannot
//!   carry effect constraints, so the callee cannot suspend between the elided retain and
//!   release.
//! - a `Parameter(entry, i)` inherits purity from every call site of its function, as a
//!   whole-program monotone fixpoint (params start pure, flip to impure);
//! - anything else -- and every parameter of a function whose value escapes callee position
//!   (unknown call sites) -- is impure.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::mir::{BlockId, Definition, DefinitionId, Instruction, InstructionId, Mir, Value};

/// Resolve `value` (within `def`) through `Instruction::Id` chains to a referenced
/// definition, when it is one.
fn resolve_definition(def: &Definition, value: &Value) -> Option<DefinitionId> {
    let mut value = value.clone();
    loop {
        match value {
            Value::Definition(d) => return Some(d),
            Value::InstructionResult(id) => match def.instructions.get(id)? {
                Instruction::Id(inner) => value = inner.clone(),
                _ => return None,
            },
            _ => return None,
        }
    }
}

/// An argument's purity classification at a call site inside `def`.
enum ArgClass {
    Pure,
    /// Inherits from the enclosing function's entry parameter `i`.
    Param(u32),
    Impure,
}

fn classify_arg(def: &Definition, value: &Value) -> ArgClass {
    if resolve_definition(def, value).is_some() {
        return ArgClass::Pure;
    }
    match value {
        Value::Parameter(block, i) if *block == BlockId::ENTRY_BLOCK => ArgClass::Param(*i),
        _ => ArgClass::Impure,
    }
}

impl Mir {
    pub(crate) fn elide_pure_borrow_pairs(mut self) -> Self {
        if self.definitions.values().all(|def| def.borrow_pairs.is_empty()) {
            return self;
        }

        // Step 1 -- escape scan: a definition whose value is used anywhere except direct
        // callee position has unknown call sites; all its parameters are impure. `Id`
        // instructions defer to their result's uses (post-mono every specialized callee is
        // reached through an Id), so escaped instruction-results push down their chains.
        let mut escaping: FxHashSet<DefinitionId> = FxHashSet::default();
        for def in self.definitions.values() {
            let mut escaped_results: FxHashSet<InstructionId> = FxHashSet::default();
            let mark = |value: &Value, escaping: &mut FxHashSet<DefinitionId>, escaped: &mut FxHashSet<InstructionId>| {
                match value {
                    Value::Definition(d) => {
                        escaping.insert(*d);
                    },
                    Value::InstructionResult(id) => {
                        escaped.insert(*id);
                    },
                    _ => (),
                }
            };
            for instruction in def.instructions.values() {
                match instruction {
                    Instruction::Call { function, arguments } => {
                        // The callee position does not escape; a non-resolvable function
                        // value is an instruction result whose own producer decides.
                        if !matches!(function, Value::Definition(_) | Value::InstructionResult(_)) {}
                        for argument in arguments {
                            mark(argument, &mut escaping, &mut escaped_results);
                        }
                    },
                    // Deferred: an Id escapes only if its result does (below).
                    Instruction::Id(_) => (),
                    other => other.for_each_value(|v| mark(v, &mut escaping, &mut escaped_results)),
                }
            }
            for (_, block) in def.blocks.iter() {
                if let Some(terminator) = &block.terminator {
                    terminator.for_each_value(|v| mark(v, &mut escaping, &mut escaped_results));
                }
            }
            // Push escapes down Id chains until stable (chains are short).
            loop {
                let mut changed = false;
                for (id, instruction) in def.instructions.iter() {
                    if let Instruction::Id(inner) = instruction
                        && escaped_results.contains(&id)
                    {
                        match inner {
                            Value::Definition(d) => changed |= escaping.insert(*d),
                            Value::InstructionResult(r) => changed |= escaped_results.insert(*r),
                            _ => (),
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Step 2 -- parameter-purity fixpoint. Params start pure; impure flips propagate
        // along caller-param → callee-param edges.
        let mut impure: FxHashSet<(DefinitionId, u32)> = FxHashSet::default();
        let mut edges: FxHashMap<(DefinitionId, u32), Vec<(DefinitionId, u32)>> = FxHashMap::default();
        let mut worklist: Vec<(DefinitionId, u32)> = Vec::new();
        for def in self.definitions.values() {
            if escaping.contains(&def.id) {
                let param_count = def.blocks.get(BlockId::ENTRY_BLOCK).map_or(0, |b| b.parameter_types.len());
                for i in 0..param_count as u32 {
                    if impure.insert((def.id, i)) {
                        worklist.push((def.id, i));
                    }
                }
            }
            for instruction in def.instructions.values() {
                let Instruction::Call { function, arguments } = instruction else { continue };
                let Some(target) = resolve_definition(def, function) else {
                    // Unknown callee: its params are handled by the escape scan (the callee
                    // value came from somewhere that marked the definition escaping).
                    continue;
                };
                for (j, argument) in arguments.iter().enumerate() {
                    match classify_arg(def, argument) {
                        ArgClass::Pure => (),
                        ArgClass::Param(i) => edges.entry((def.id, i)).or_default().push((target, j as u32)),
                        ArgClass::Impure => {
                            if impure.insert((target, j as u32)) {
                                worklist.push((target, j as u32));
                            }
                        },
                    }
                }
            }
        }
        while let Some(fact) = worklist.pop() {
            if let Some(dependents) = edges.get(&fact) {
                for dependent in dependents.clone() {
                    if impure.insert(dependent) {
                        worklist.push(dependent);
                    }
                }
            }
        }

        // Step 3 -- elide pairs whose implicit arguments are all pure under the fixpoint.
        for def in self.definitions.values_mut() {
            if def.borrow_pairs.is_empty() {
                continue;
            }
            let live: FxHashSet<InstructionId> =
                def.blocks.iter().flat_map(|(_, b)| b.instructions.iter().copied()).collect();
            let mut removals: FxHashSet<InstructionId> = FxHashSet::default();
            let pairs = std::mem::take(&mut def.borrow_pairs);
            for pair in pairs {
                // A pair whose instructions were rewritten away (dead bodies, effect
                // splices) is stale; skip it defensively.
                if !live.contains(&pair.retain) || !live.contains(&pair.release_call) || !live.contains(&pair.call) {
                    continue;
                }
                let Some(Instruction::Call { arguments, .. }) = def.instructions.get(pair.call) else { continue };
                let pure = pair.implicit_args.iter().all(|&j| match arguments.get(j as usize) {
                    Some(argument) => match classify_arg(def, argument) {
                        ArgClass::Pure => true,
                        ArgClass::Param(i) => !impure.contains(&(def.id, i)),
                        ArgClass::Impure => false,
                    },
                    None => false,
                });
                if pure {
                    removals.insert(pair.retain);
                    removals.insert(pair.release_call);
                }
            }
            if !removals.is_empty() {
                for (_, block) in def.blocks.iter_mut() {
                    block.instructions.retain(|id| !removals.contains(id));
                }
            }
        }

        self
    }
}
