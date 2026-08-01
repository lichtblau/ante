#!/usr/bin/env bash
# Stage 0 sanitizer lane (dropping/05-stage0-harness.md, deliverable 2).
#
# Compiles and runs every examples/codegen/**/*.an through the C backend with
# AddressSanitizer + LeakSanitizer, and classifies each as CLEAN / LEAK / ERROR.
# The C backend shells out to `cc`; a shim first on PATH appends
# `-fsanitize=address`, so ASan instruments the whole program. The compiler also
# *runs* the produced binary, so the sanitizer fires during the `ante` invocation.
#
# Each source is copied to a temp build dir before compiling, so no .c/.o/binary
# artifacts ever land in the repo (Std.* imports resolve via the compiler, not the
# cwd, so location does not matter).
#
# Exit status:
#   - nonzero if any ERROR (double-free / use-after-free / overflow) — ALWAYS.
#   - nonzero if any LEAK — only when --gate-leaks is passed (leaks are expected
#     until Stage 1 lands; the flag flips on in Stage 1e).
#   - BUILD-FAIL (a source that does not compile) is reported but never gates.
#
# Usage:
#   tests/asan/run.sh [--gate-leaks] [--quiet] [PATH ...]
#   (PATH defaults to examples/codegen)
#
# Requires: a C-backend compiler at target/debug/ante
#   (cargo build --no-default-features   # LLVM is not needed for the C backend)
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ante="$repo/target/debug/ante"

gate_leaks=0
quiet=0
roots=()
for arg in "$@"; do
    case "$arg" in
        --gate-leaks) gate_leaks=1 ;;
        --quiet) quiet=1 ;;
        *) roots+=("$arg") ;;
    esac
done
[[ ${#roots[@]} -eq 0 ]] && roots=("$repo/examples/codegen")

if [[ ! -x "$ante" ]]; then
    echo "error: $ante not found. Build with: cargo build --no-default-features" >&2
    exit 2
fi

# Load the known pre-existing-error allowlist (repo-relative paths, one per line).
declare -A known_error
known_file="$repo/tests/asan/known_errors.txt"
if [[ -f "$known_file" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line//[[:space:]]/}"
        [[ -n "$line" ]] && known_error["$line"]=1
    done < "$known_file"
fi

# Known remaining leaks (Stage 2-4 residue classes); --gate-leaks fails only on
# files NOT listed here.
declare -A known_leak
known_leaks_file="$repo/tests/asan/known_leaks.txt"
if [[ -f "$known_leaks_file" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line//[[:space:]]/}"
        [[ -n "$line" ]] && known_leak["$line"]=1
    done < "$known_leaks_file"
fi

# ASan cc shim.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/shim" "$tmp/build"
cat > "$tmp/shim/cc" <<'SH'
#!/usr/bin/env bash
exec /usr/bin/cc -fsanitize=address -fno-omit-frame-pointer -g "$@"
SH
chmod +x "$tmp/shim/cc"
export PATH="$tmp/shim:$PATH"
export ASAN_OPTIONS="detect_stack_use_after_return=1:detect_leaks=1:exitcode=23"

n_clean=0; n_leak=0; n_known_leak=0; n_error=0; n_known_err=0; n_build=0; leak_total=0

# Collect files (sorted, stable).
mapfile -t files < <(for r in "${roots[@]}"; do find "$r" -name '*.an' -type f; done | sort)

for f in "${files[@]}"; do
    rel="${f#$repo/}"
    src="$tmp/build/$(basename "$f")"
    cp "$f" "$src"
    out="$(cd "$tmp/build" && timeout 60 "$ante" --backend c --delete-binary --no-color "$src" 2>&1)"
    rm -f "$tmp/build/"*.c "$tmp/build/"*.o "$src"

    if printf '%s' "$out" | grep -qE 'AddressSanitizer: (heap-use-after-free|stack-use-after-return|heap-buffer-overflow|stack-buffer-overflow|global-buffer-overflow|attempting double-free|bad-free|alloc-dealloc-mismatch|SEGV|use-after-poison)'; then
        kind="$(printf '%s' "$out" | grep -oE 'AddressSanitizer: [a-z-]+' | head -1)"
        if [[ -n "${known_error[$rel]:-}" ]]; then
            n_known_err=$((n_known_err+1))
            [[ $quiet -eq 0 ]] && printf '  KNOWN-ERR %-50s %s\n' "$rel" "$kind"
        else
            n_error=$((n_error+1))
            [[ $quiet -eq 0 ]] && printf '  ERROR   %-52s %s\n' "$rel" "$kind"
        fi
    elif printf '%s' "$out" | grep -qE 'LeakSanitizer: detected memory leaks|byte\(s\) leaked'; then
        bytes="$(printf '%s' "$out" | grep -oE '[0-9]+ byte\(s\) leaked' | grep -oE '^[0-9]+' | head -1)"
        bytes="${bytes:-0}"
        leak_total=$((leak_total+bytes))
        if [[ -n "${known_leak[$rel]:-}" ]]; then
            n_known_leak=$((n_known_leak+1))
            [[ $quiet -eq 0 ]] && printf '  KNOWN-LEAK %-49s %s bytes\n' "$rel" "$bytes"
        else
            n_leak=$((n_leak+1))
            [[ $quiet -eq 0 ]] && printf '  LEAK    %-52s %s bytes\n' "$rel" "$bytes"
        fi
    elif printf '%s' "$out" | grep -qE '^error:|Found [0-9]+ error|panicked at'; then
        n_build=$((n_build+1))
        [[ $quiet -eq 0 ]] && printf '  BUILD?  %-52s (does not compile)\n' "$rel"
    else
        n_clean=$((n_clean+1))
        [[ $quiet -eq 0 ]] && printf '  clean   %-52s\n' "$rel"
    fi
done

echo
echo "==== sanitizer lane summary ===="
printf 'total %d | clean %d | NEW-leak %d | known-leak %d (%d bytes total) | ERROR %d | known-err %d | build-fail %d\n' \
    "${#files[@]}" "$n_clean" "$n_leak" "$n_known_leak" "$leak_total" "$n_error" "$n_known_err" "$n_build"

if [[ $n_error -gt 0 ]]; then
    echo "FAIL: $n_error NEW memory error(s) detected (not in tests/asan/known_errors.txt)."
    exit 1
fi
if [[ $gate_leaks -eq 1 && $n_leak -gt 0 ]]; then
    echo "FAIL (--gate-leaks): $n_leak NEW leaking test(s) (not in tests/asan/known_leaks.txt)."
    exit 1
fi
echo "OK: no NEW memory errors ($n_known_err known pre-existing)$([[ $gate_leaks -eq 0 ]] && echo "; leaks not gated")."
exit 0
