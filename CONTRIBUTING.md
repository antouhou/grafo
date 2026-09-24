# Coding style and best practices

## Structure of the project
The project consist of several modules that shouldn't mix responsibilities with each other:
- [`src/core`]: all basic math and geometry structures; If you need to use a math structure such as rect, color, color
conversion function, CPU representation of a tesellated vertex - look into core module. If you need to introduce a new
math/geometry structure, add it to the core. But first, make sure that the structure or helper are not yet present in
some shape or form in the core module.
- [`src/commands`]

## Basic rules

- Use full, descriptive names for variables, functions, types, and modules. Do not abbreviate them. NEVER shorten the names, as it makes it extremely hard to read and review.
- Keep functions small, avoid deep nesting, and document behavior that a new contributor could miss. The new developer onboarding to the project should be able to easily grasp what the code does and why.
- Group all `use` imports at the top of the file.
- Do not define structs/enums/traits inside function bodies. Keep types at module/file scope unless there is a compelling, strictly local reason.
- Avoid `unwrap` except in tests or when failure is impossible, such as after checking `Option::is_some`. Use `?` to propagate errors or handle them where they occur. Log errors with `tracing::error!` or `tracing::warn!`. Reserve panics for critical, unrecoverable errors.
- When fixing a bug, write a failing regression test first, then fix the bug, and rerun the test.
- Put unit tests and their helpers in `foo/tests.rs`, alongside `foo/mod.rs`.
- Avoid dynamic dispatch wherever possible. Use generics and `impl Trait` instead of `Box<dyn Trait>`/`Arc<dyn Trait>`.
- All e2e tests must be as watertight as possible.
- Do not use global state whatsoever. It makes tests depend on shared state and execution order.
- Run `cargo test --workspace --all-features`.
- Run `cargo clippy --fix --all-features --tests -- -D warnings` and fix all warnings.
- Run `cargo fmt --all` to format the code after making changes.
- Avoid pyramids of doom and deep nesting, break down things into small, readable functions.
- Keep comments short. Explain behavior that the code does not make clear. Omit plans and thought processes.
- Use `thiserror` and derive `thiserror::Error` for error types.

## Project-specific guidelines

- Avoid heap allocations in the render loop. Reuse allocated storage where possible.
- Always run `tests/visual_regression.rs` to check for regressions.
- Cover every combination of a new feature with existing features in the visual regression tiles. Use the existing tiles as examples.
- For a rendering bug, add a tile covering the combination that caused it. Create a separate test only when a tile cannot cover the behavior.

## Test organization

- Do not put tests in the same file as implementation;
- Do not put test helpers next to implementation - put them into test file;
- Split modules into directories when they should contain tests: mod.rs for the implementation and tests.rs for tests;
- If the module doesn't need testing, do not create a directory for the module;
- Do not add tests that simply check setters or trivial functionality;
- Do not add tests that require a GPU. That's because my CI doesn't have a GPU right now; There's only exception to this
rule: visual regression test scene. I run it manually to check that everything is fine.
