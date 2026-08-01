use crate::mir::DefinitionId;

/// A global that cannot be initialized by a constant C initializer (it reads another global's
/// value). Its definition is emitted zero-initialized; `statement` assigns the real value at
/// startup. `deps` names the other deferred globals that must be assigned first.
pub(super) struct GlobalInitializer {
    pub id: DefinitionId,
    pub deps: Vec<DefinitionId>,
    pub statement: String,
}

/// The output .c file.
/// This file is divided into several sections to ensure everything is declared before it is used.
#[derive(Default)]
pub(super) struct CFile {
    includes: String,
    type_declarations: String,
    type_definitions: String,
    function_declarations: String,
    global_declarations: String,
    global_definitions: String,
    function_definitions: String,

    /// Runtime assignments for globals with non-constant initializers, run before `main`.
    global_initializers: Vec<GlobalInitializer>,
}

impl CFile {
    /// Consume the file, concatenating everything into a single contents string
    pub(super) fn into_contents(self) -> String {
        let mut result = self.includes;
        result.reserve_exact(
            self.type_declarations.len()
                + self.type_definitions.len()
                + self.function_declarations.len()
                + self.global_declarations.len()
                + self.global_definitions.len()
                + self.function_definitions.len()
                + 6, // 6 for the newlines separating each.
        );
        let capacity = result.capacity();
        result += "\n";
        result += &self.type_declarations;
        result += "\n";
        result += &self.type_definitions;
        result += "\n";
        result += &self.function_declarations;
        result += "\n";
        result += &self.global_declarations;
        result += "\n";
        result += &self.global_definitions;
        result += "\n";
        result += &self.function_definitions;
        // Ensure the capacity estimate was correct
        assert_eq!(capacity, result.capacity());
        result
    }

    // TODO: Should [super::write_cached_tuple_type] be changed to use this method?
    #[allow(unused)]
    pub(super) fn add_type_declaration(&mut self, decl: &str) {
        self.type_declarations += decl;
        self.type_declarations += "\n";
    }

    pub(super) fn add_type_definition(&mut self, def: &str) {
        self.type_definitions += def;
        self.type_definitions += "\n";
    }

    pub(super) fn add_function_declaration(&mut self, decl: &str) {
        self.function_declarations += decl;
        self.function_declarations += "\n";
    }

    pub(super) fn add_global_declaration(&mut self, decl: &str) {
        self.global_declarations += decl;
        self.global_declarations += "\n";
    }

    pub(super) fn add_global_definition(&mut self, def: &str) {
        self.global_definitions += def;
        self.global_definitions += "\n";
    }

    pub(super) fn add_function_definition(&mut self, def: &str) {
        self.function_definitions += def;
        self.function_definitions += "\n";
    }

    pub(super) fn add_global_initializer(&mut self, init: GlobalInitializer) {
        self.global_initializers.push(init);
    }

    /// Take the collected non-constant global initializers, leaving the list empty. Consumed by
    /// [super::build_c_file] to emit a startup function rather than written into `into_contents`.
    pub(super) fn take_global_initializers(&mut self) -> Vec<GlobalInitializer> {
        std::mem::take(&mut self.global_initializers)
    }

    /// Extend `self` with the contents of `other`
    pub(super) fn extend(mut self, other: CFile) -> CFile {
        self.includes += &other.includes;
        self.type_declarations += &other.type_declarations;
        self.type_definitions += &other.type_definitions;
        self.function_declarations += &other.function_declarations;
        self.global_declarations += &other.global_declarations;
        self.global_definitions += &other.global_definitions;
        self.function_definitions += &other.function_definitions;
        self.global_initializers.extend(other.global_initializers);
        self
    }

    /// Add some necessary items to this CFile that are needed by all Ante programs:
    /// the standard headers and runtime prototypes the generated code references, plus the
    /// `Unit` struct.
    pub(crate) fn add_starter_items(mut self) -> Self {
        // stdlib.h, string.h, math.h would conflict with `extern` statements in source code
        // which declare some of the same functions.
        self.includes += "#include <stdint.h>\n";
        self.includes += "#include <stddef.h>\n";
        self.includes += "#include <stdbool.h>\n";

        self.includes += "\
#if defined(__FLT32_MANT_DIG__) && defined(__FLT64_MANT_DIG__)
typedef _Float32 ante_f32;
typedef _Float64 ante_f64;
#else
typedef float ante_f32;
typedef double ante_f64;
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ANTE_UNREACHABLE() __builtin_unreachable()
#define ANTE_INF() __builtin_inf()
#define ANTE_NAN() __builtin_nan(\"\")
#elif defined(_MSC_VER)
#include <math.h>
#define ANTE_UNREACHABLE() __assume(0)
#define ANTE_INF() INFINITY
#define ANTE_NAN() NAN
#else
#include <math.h>
#define ANTE_UNREACHABLE() ((void)0)
#define ANTE_INF() INFINITY
#define ANTE_NAN() NAN
#endif
";

        self.type_declarations += "typedef struct { char _unused; } Unit;\n";

        // Shared heap allocations carry a refcount header immediately before the
        // value. `AllocShared` allocates `{header, value}` and returns a pointer AT `value`; the
        // header is a max-aligned `size_t` so `value` keeps malloc's alignment and every
        // load/GEP/store through the pointer is byte-identical. Only `FreeShared` knows the offset:
        // it frees `value_ptr - ANTE_RC_HEADER_SIZE`.  0-arg shared constructors' backing statics
        // get `count = 0` (the immortal sentinel).
        //
        // `ANTE_RC_HEADER_SIZE` must equal `sizeof(AnteRcHeader)`, not `sizeof(max_align_t)`: the
        // immortal-sentinel backing static is laid out as `struct { AnteRcHeader _hdr; T value; }`,
        // so `value` sits at `sizeof(AnteRcHeader)` -- and on platforms where `sizeof(max_align_t)`
        // exceeds the header struct's size (align-16 header but a 32-byte `max_align_t`), the two
        // disagree and retain/decrement read the count out of bounds. Keeping the offset tied to
        // the struct keeps heap allocations and statics byte-identical.

        self.type_declarations += "typedef struct { _Alignas(max_align_t) size_t count; } AnteRcHeader;\n";
        self.type_declarations += "#define ANTE_RC_HEADER_SIZE (sizeof(AnteRcHeader))\n";

        // The count a block wears between "the last reference went away" and `FreeShared`. During
        // that window a user `Drop` impl is running, and it can still see the handle -- so it can
        // copy it somewhere that outlives the free. The count cannot say so on its own: a plain 0
        // is the immortal static sentinel, which retain deliberately ignores. The dying block
        // gets its own value instead, so a retain during teardown is recognizable, and fatal:
        // resurrection is a use-after-free the moment the block is freed a few instructions later.
        self.type_declarations += "#define ANTE_RC_DYING ((size_t)-1)\n";

        self.function_declarations += "void* malloc(size_t);\n";
        // `free` is declared here (matching the stdlib FFI's `Unit free(void*)`) so `FreeShared`
        // can call it even when the program imports no `Std.C.free` of its own.
        self.function_declarations += "Unit free(void*);\n";
        self.function_declarations += "void* memcpy(void*, void*, size_t);\n";
        self.function_declarations += "void* memset(void*, int, size_t);\n";
        self.function_declarations += "double fmod(double, double);\n";

        // Declared rather than included (stdio.h/stdlib.h would clash with source `extern`s), and
        // deliberately not through `puts`/`fputs`/`fwrite`: `Std.C` binds those under their real C
        // names, so declaring them here is a conflicting redeclaration for any program that imports
        // them. `write`, `fflush` and `abort` it does not bind.
        //
        // The message goes to fd 2 (unbuffered), but `abort` discards whatever the program itself
        // has buffered on stdout -- including the output of the `Drop` impl that just ran, which is
        // the context that makes the message legible. So flush every stream first.
        self.function_declarations += "long write(int, const void*, unsigned long);\n";
        self.function_declarations += "int fflush(void*);\n";
        self.function_declarations += "void abort(void);\n";
        self.function_declarations += "\
static void ante_rc_resurrected(void) {
    static const char ante_rc_msg[] = \"ante: a shared value was resurrected: its refcount reached \
zero and its Drop impl copied the handle back out. The block is freed when the impl returns, so the \
copy would dangle.\\n\";
    fflush(0);
    write(2, ante_rc_msg, sizeof(ante_rc_msg) - 1);
    abort();
}
";
        self
    }
}
