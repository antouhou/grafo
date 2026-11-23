use std::process::Command;

#[test]
fn renderer_builds_for_wasm() {
    let build_status = Command::new(env!("CARGO"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "build",
            "--locked",
            "--lib",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
            "target/wasm-build-test",
        ])
        .status()
        .expect("failed to start Cargo");

    assert!(build_status.success(), "WASM build failed: {build_status}");
}
