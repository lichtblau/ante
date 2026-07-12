#!/usr/bin/env bash
# Builds a fully static (musl) release binary of the C-backend `ante` compiler.
#
# The `llvm` feature is dropped because there is no musl-built libLLVM
# available (apt's LLVM packages are glibc-only, and building LLVM 21 from
# source against musl is its own project), so this only produces the
# C-backend build. The `ante` binary itself ends up statically linked with
# no runtime library dependencies, but it still shells out to the system
# `cc` to compile/link Ante programs (including aminicoro/minicoro.c), so a
# C toolchain must still be present on any machine that *runs* it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# GCC < 14 doesn't special-case `__has_feature` as a reserved feature-check
# form (only clang and GCC 14+ do), so on it
# `defined(__has_feature) && __has_feature(x)` substitutes the plain,
# undefined identifier down to `0(0)`, a hard syntax error. ubuntu-latest's
# `cc` is still GCC 13. Patch it out at build time instead of touching the
# aminicoro submodule/fork.
if ! grep -q '__has_feature(x) 0' aminicoro/minicoro.c; then
    patch -p0 aminicoro/minicoro.c <<'EOF'
--- minicoro.c
+++ minicoro.c
@@ -189,7 +189,10 @@
     longjmp(ctx->jb, val);
 }

-#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
+#ifndef __has_feature
+# define __has_feature(x) 0
+#endif
+#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)
 void __asan_unpoison_memory_region(void const volatile* addr, size_t size);
 #endif

EOF
fi

TARGET=x86_64-unknown-linux-musl

rustup target add "$TARGET"
cargo build --release --no-default-features --target "$TARGET"

BIN="target/$TARGET/release/ante"
strip "$BIN"

echo "Static musl binary: $BIN"
