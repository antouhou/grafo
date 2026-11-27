#!/usr/bin/env bash
# Run the release benchmark and copy its first completed result to the host.
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
results_directory=${1:-ui-bench-results}
mkdir -p -- "$results_directory"
results_directory=$(cd -- "$results_directory" && pwd)
printf 'Building release UI bench for %s; requesting vsync off by default.\n' "$abi"
cargo apk build -p ui-bench --lib --release --target "$target"
cargo_metadata=$(cargo metadata --no-deps --format-version 1)
target_directory=$(jq -er '.target_directory' <<< "$cargo_metadata")
apk="$target_directory/release/apk/ui-bench.apk"
adb install -r "$apk"
package=org.grafo.ui_bench
adb shell am force-stop "$package"
run_started=$(adb shell 'date "+%Y-%m-%d %H:%M:%S.000"')
run_started=${run_started//$'\r'/}
adb shell am start -W -n "$package/android.app.NativeActivity"
if ! pid=$(adb shell pidof "$package"); then
    printf 'The benchmark exited before result collection could start.\n' >&2
    exit 1
fi
pid=${pid//$'\r'/}
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    printf 'Could not find the benchmark process.\n' >&2
    exit 1
fi

printf 'Waiting for a completed benchmark run.\n'
logcat_pid=
download=
cleanup() {
    if [[ -n "$logcat_pid" ]]; then
        kill "$logcat_pid" 2>/dev/null || true
        wait "$logcat_pid" 2>/dev/null || true
    fi
    if [[ -n "$download" ]]; then
        rm -f -- "$download"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec 3< <(exec adb logcat -b main -b events -v brief -T "$run_started" \
    'ui-bench:I' 'am_proc_died:I' 'am_kill:I' '*:S')
logcat_pid=$!
application_pattern="^[VDIWEF]/ui-bench[[:space:]]*\\([[:space:]]*$pid[[:space:]]*\\):[[:space:]]*(.*)$"
death_pattern="^[VDIWEF]/am_(proc_died|kill)[[:space:]]*\\([[:space:]]*[0-9]+\\):[[:space:]]*\\[[0-9]+,[[:space:]]*$pid,"
device_result=
while IFS= read -r notification <&3; do
    notification=${notification//$'\r'/}
    if [[ "$notification" =~ $application_pattern ]]; then
        message=${BASH_REMATCH[1]}
        case "$message" in
            *"UI_BENCH_RESULT "*) device_result=${message##*UI_BENCH_RESULT }; break ;;
            *UI_BENCH_FAILED*|*UI_BENCH_EXITED*)
                printf 'No completed result was saved: %s\n' "$message" >&2
                exit 1
                ;;
        esac
    elif [[ "$notification" =~ $death_pattern ]]; then
        printf 'The benchmark process ended before saving a result: %s\n' "$notification" >&2
        exit 1
    fi
done
exec 3<&-
kill "$logcat_pid" 2>/dev/null || true
wait "$logcat_pid" 2>/dev/null || true
logcat_pid=
if [[ -z "$device_result" ]]; then
    printf 'The Android log stream ended before a result was saved.\n' >&2
    exit 1
fi
filename=${device_result##*/}
case "$filename" in
    run_*.json) ;;
    *) printf 'Unexpected result filename: %s\n' "$filename" >&2; exit 1 ;;
esac
destination="$results_directory/$filename"
if [[ -e "$destination" ]]; then
    printf 'Result already exists on the host: %s\n' "$destination" >&2
    exit 1
fi
download=$(mktemp "$results_directory/.ui-bench-download.XXXXXX")
adb pull "$device_result" "$download"
jq -e . "$download" > /dev/null
mv -n -- "$download" "$destination"
if [[ -e "$download" ]]; then
    printf 'Result already exists on the host: %s\n' "$destination" >&2
    exit 1
fi
printf 'Saved benchmark result: %s\n' "$destination"
