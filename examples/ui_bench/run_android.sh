#!/usr/bin/env bash
# Build, install, and launch the release benchmark for the connected device's ABI.
set -euo pipefail

abi=$(adb shell getprop ro.product.cpu.abi)
abi=${abi//$'\r'/}
case "$abi" in
    armeabi-v7a) target=armv7-linux-androideabi ;;
    arm64-v8a) target=aarch64-linux-android ;;
    x86_64) target=x86_64-linux-android ;;
    x86) target=i686-linux-android ;;
    *) printf 'Unsupported Android ABI: %s\n' "$abi" >&2; exit 1 ;;
esac

export CARGO_APK_RELEASE_KEYSTORE=${CARGO_APK_RELEASE_KEYSTORE-"$HOME/.android/debug.keystore"}
export CARGO_APK_RELEASE_KEYSTORE_PASSWORD=${CARGO_APK_RELEASE_KEYSTORE_PASSWORD-android}
if [[ ! -f "$CARGO_APK_RELEASE_KEYSTORE" ]]; then
    printf 'No signing keystore. See the benchmark README for the one-time keytool command.\n' >&2
    exit 1
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
printf 'Building release UI bench for %s; requesting vsync off by default.\n' "$abi"
cargo apk build -p ui-bench --lib --release --target "$target"
cargo_metadata=$(cargo metadata --no-deps --format-version 1)
target_directory=$(jq -er '.target_directory' <<< "$cargo_metadata")
apk="$target_directory/release/apk/ui-bench.apk"
adb install -r "$apk"
package=org.grafo.ui_bench
adb shell am force-stop "$package"
adb shell am start -W -n "$package/android.app.NativeActivity"
