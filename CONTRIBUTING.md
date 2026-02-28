# Coding style and best practices

## Basic rules:
- Use full, descriptive names for variables, functions, types, and modules. NEVER shorten the names, as it makes it extremely hard to read and review.
- Functions should be small and avoid deep nesting, well-structured and documented. The new developer onboarding to the project should be able to easily grasp what the code does and why.
- Group all `use` imports at the top of the file (no local import islands). Prefer ordering: std → third-party → workspace crates.
- Do not define structs/enums/traits inside function bodies. Keep types at module/file scope unless there is a compelling, strictly local reason.
- Use of `unwrap` is discouraged except in tests or in cases where failure is impossible (e.g., after checking `Option::is_some`). Prefer using `?` to propagate errors or handle them gracefully. Log errors using tracing crate macros (e.g., `tracing::error!`, `tracing::warn!`) instead of panicking, unless it's a critical, unrecoverable error.
- When fixing a bug, first write a test that reproduces the bug, then fix the bug, and finally ensure the test passes.
- Avoid dynamic dispatch wherever possible. Use generics and `impl Trait` instead of `Box<dyn Trait>`/`Arc<dyn Trait>`.
- All e2e tests must be as watertight as possible.
- No global state whatsoever. It makes reliable and quick testing a very hard task.
- Run `cargo clippy --fix --all-features --tests -D warnings` to ensure there are no clippy warnings.
- Run `cargo fmt --all` to format the code automatically after you've done necessary changes.
- Avoid pyramids of doom and deep nesting, break down things into small, readable functions.
- Make sure comments are short and descriptive, don't contain the thought process or plans - they should strictly clarify parts that might be unobvious from surrounding code, nothing more.
- Use `thiserror` and derive `thiserror::Error` for error types.

## Project-specific guidelines:
- Avoid heap allocations in the render method/loop. Try to reuse values allocated on the heap as much as possible in general.
- `tests/visual_regression.rs` always must be run to check that no regression has been introduced.
- In case of adding a new feature, expand `tests/visual_regression.rs` to test that it works as expected, with all possible permutations of the added feature and existing ones. Study other cases in the test for reference.
  In case when a new bug is found, make sure it is fixed, and add the permutation that caused it to `tests/visual_regression.rs`.