#!/usr/bin/env python3
"""Build, install, and launch the release benchmark for the connected device's ABI."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="ADB serial, needed when several devices are connected")
    arguments = parser.parse_args()
    for command in ("cargo", "adb"):
        if shutil.which(command) is None:
            parser.error(f"{command} is not on PATH")

    adb = ["adb"]
    device = arguments.device or os.environ.get("ANDROID_SERIAL")
    if device:
        adb += ["-s", device]
    abi = subprocess.check_output(adb + ["shell", "getprop", "ro.product.cpu.abi"], text=True).strip()
    targets = {
        "armeabi-v7a": "armv7-linux-androideabi",
        "arm64-v8a": "aarch64-linux-android",
        "x86_64": "x86_64-linux-android",
        "x86": "i686-linux-android",
    }
    if abi not in targets:
        parser.error(f"Unsupported Android ABI: {abi}")
    environment = os.environ.copy()
    keystore = environment.setdefault("CARGO_APK_RELEASE_KEYSTORE", str(Path.home() / ".android" / "debug.keystore"))
    environment.setdefault("CARGO_APK_RELEASE_KEYSTORE_PASSWORD", "android")
    if not Path(keystore).is_file():
        parser.error("No signing keystore. See the benchmark README for the one-time keytool command.")
    root = Path(__file__).resolve().parents[2]
    command = ["cargo", "apk", "build", "-p", "grafo-launcher-benchmark", "--lib", "--release", "--target", targets[abi]]
    print(f"Building release benchmark for {abi}; vsync is off by default.", flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=root, env=environment, text=True
    ))
    apk = Path(metadata["target_directory"]) / "release" / "apk" / "grafo-launcher-benchmark.apk"
    subprocess.run(adb + ["install", "-r", str(apk)], check=True)
    package = "org.grafo.launcher_benchmark"
    subprocess.run(adb + ["shell", "am", "force-stop", package], check=True)
    subprocess.run(adb + ["shell", "am", "start", "-W", "-n", f"{package}/android.app.NativeActivity"], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
