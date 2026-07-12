# Bare image: nothing but the statically linked ante binary and the
# precompiled minicoro.o it's built to look for at /minicoro.o (see
# ANTE_MINICORO_PATH in .github/workflows/release.yml). No libc, no shell,
# no stdlib, no C compiler - this is a distribution artifact, not a
# runnable dev environment.
FROM scratch
COPY target/x86_64-unknown-linux-musl/release/ante /ante
COPY minicoro.o /minicoro.o
ENTRYPOINT ["/ante"]
