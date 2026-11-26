# UI bench

Automatic Grafo UI benchmark. Vsync off by default.

This bench assesses the most basic UI rendering performance. It is suggested to run it on a separate device to avoid
any possible interference, preferably an Android TV box, so the regressions are easier to spot.

Run from the repository root:

```sh
# Desktop
cargo run --release -p ui-bench

# Android
bash examples/ui_bench/run_android.sh
adb logcat -s ui-bench
```

Android needs Bash, jq, an authorized device, and
[cargo-apk setup](https://github.com/rust-mobile/cargo-apk), including SDK/NDK and the device's Rust target.

If `~/.android/debug.keystore` is missing, create it once:

```sh
keytool -genkeypair -keystore ~/.android/debug.keystore -storepass android -keypass android -alias androiddebugkey -dname "CN=Android Debug,O=Android,C=US" -keyalg RSA -keysize 2048 -validity 10000
```

[Font notices](assets/NOTICE.md).
