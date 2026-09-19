# Coding style and best practices

## Basic rules

- Use full, descriptive names for variables, functions, types, and modules. Do not abbreviate them. NEVER shorten the names, as it makes it extremely hard to read and review.
- Keep functions small, avoid deep nesting, and document behavior that a new contributor could miss. The new developer onboarding to the project should be able to easily grasp what the code does and why.
- Group all `use` imports at the top of the file.
- Do not define structs/enums/traits inside function bodies. Keep types at module/file scope unless there is a compelling, strictly local reason.
- Avoid `unwrap` except in tests or when failure is impossible, such as after checking `Option::is_some`. Use `?` to propagate errors or handle them where they occur. Log errors with `tracing::error!` or `tracing::warn!`. Reserve panics for critical, unrecoverable errors.
- When fixing a bug, write a failing regression test first, then fix the bug, and rerun the test.
- Avoid dynamic dispatch wherever possible. Use generics and `impl Trait` instead of `Box<dyn Trait>`/`Arc<dyn Trait>`.
- All e2e tests must be as watertight as possible.
- Do not use global state whatsoever. It makes tests depend on shared state and execution order.
- Run `cargo clippy --fix --all-features --tests -D warnings` and fix all warnings.
- Run `cargo fmt --all` to format the code after making changes.
- Avoid pyramids of doom and deep nesting, break down things into small, readable functions.
- Keep comments short. Explain behavior that the code does not make clear. Omit plans and thought processes.
- Use `thiserror` and derive `thiserror::Error` for error types.

## Project-specific guidelines

- Avoid heap allocations in the render loop. Reuse allocated storage where possible.
- Always run `tests/visual_regression.rs` to check for regressions.
- Cover every combination of a new feature with existing features in the visual regression tiles. Use the existing tiles as examples.
- For a rendering bug, add a tile covering the combination that caused it. Create a separate test only when a tile cannot cover the behavior.
