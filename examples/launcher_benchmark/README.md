# Launcher benchmark

Automatic Grafo UI benchmark. Vsync off by default.

Run from the repository root:

```sh
# Desktop
cargo run --release -p grafo-launcher-benchmark

# Android
bash examples/launcher_benchmark/run_android.sh
adb logcat -s grafo-benchmark
```

Android needs Bash, jq, an authorized device, and
[cargo-apk setup](https://github.com/rust-mobile/cargo-apk), including SDK/NDK and the device's Rust target.

If `~/.android/debug.keystore` is missing, create it once:

```sh
keytool -genkeypair -keystore ~/.android/debug.keystore -storepass android -keypass android -alias androiddebugkey -dname "CN=Android Debug,O=Android,C=US" -keyalg RSA -keysize 2048 -validity 10000
```

[Font notices](assets/NOTICE.md).
