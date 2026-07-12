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
cd "$(dirname "${BASH_SOURCE[0]}")"

TARGET=x86_64-unknown-linux-musl

rustup target add "$TARGET"
cargo build --release --no-default-features --target "$TARGET"

BIN="target/$TARGET/release/ante"
strip "$BIN"

echo "Static musl binary: $BIN"
