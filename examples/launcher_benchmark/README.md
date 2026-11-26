# Launcher benchmark

A retained Grafo interface inspired by Momo's app grid and quick settings. It runs
page slides, settings open/close, settings scrolling, and power-menu transitions
automatically, with separate timing results for each. It has no blur passes.

The scene has three pages of app icons and labels, nested rounded clips, overlapping
icon artwork, a moving page indicator, and a sliding settings panel with clipped
tiles and an unblurred shape-mask shadow. The normal view has three large
app cards per page, like Momo's TV layout. The stress mode has 18 icons per page.
Page transitions take 180 ms, settings transitions take 280 ms, and power-menu
transitions take 240 ms.

This estimates the renderer's workload. It does not include Momo's component,
layout, IPC, or application logic, so its FPS is not a prediction of Momo's FPS.

## Settings workloads

The scroll workload moves eight tiles, their vector icons, and text through a fixed
rectangular viewport inside the rounded panel. The sliders stay in place. Fractional
scroll positions change path coverage and intersections with nested clips.

The power workload creates the incoming menu, slides both menus in opposite
directions, and removes the outgoing menu at the end. During the slide, an extra
rectangular viewport clips both menus while the rounded panel shrinks or grows.
The panel's changing geometry also changes its shadow-mask dimensions. Panel
heights are prepared in half-pixel steps at startup; motion reuses those shapes.
Menu insertion and removal are measured, including the final stationary draw.
The power buttons are decorative and never invoke system actions.

These cases follow Momo's `Scrollable` and `ViewTransition` behavior. Device
profiling found substantial CPU time in clipping and coverage generation, alongside
Daiko's component updates. A control run on the G31, with Momo's shadow removed,
still reached 59 ms per renderer call during power transitions. Holding panel
height fixed during the slide reduced the sampled maximum to 21 ms. This points
to changing clip geometry as a contributor beyond the shadow itself.

Rounded surfaces use Grafo's native rounded rectangles. Visible vector icons use
general paths. Momo currently passes its rounded boxes through Daiko as general
paths, so this benchmark intentionally preserves Grafo's faster rounded-rectangle
handling. It also excludes Daiko layout and shape reconstruction costs. Compare
the exercised rendering behavior, rather than expecting identical FPS.

## Desktop

From the repository root, on macOS, Windows, or Linux:

```sh
cargo run --release -p grafo-launcher-benchmark
```

Use a release build for comparisons. Linux needs the usual winit X11/Wayland
development dependencies. The initial window is 1280 x 720 physical pixels. Resize
it or use fullscreen to match the Android output resolution. The scene scales
uniformly from a 1280 x 720 layout; landscape is recommended.

## Android

Install the Android SDK, NDK, Rust target, and cargo-apk once. Configure the SDK/NDK
environment as described in the [cargo-apk instructions](https://github.com/rust-mobile/cargo-apk).
For the connected 32-bit G31 device:

```sh
rustup target add armv7-linux-androideabi
cargo install cargo-apk
```

For a 64-bit phone, install `aarch64-linux-android` instead. Enable USB debugging,
authorize the computer, and check that `adb devices` lists the device.

The launcher uses the standard Android debug signing key for local release builds.
If that key does not exist, create it once:

```sh
keytool -genkeypair -keystore ~/.android/debug.keystore -storepass android -keypass android -alias androiddebugkey -dname "CN=Android Debug,O=Android,C=US" -keyalg RSA -keysize 2048 -validity 10000
```

Then run from the repository root:

```sh
python3 examples/launcher_benchmark/run_android.py
```

On Windows, use `py` instead of `python3` and the Windows home path in the keystore
command. The script detects the device's ABI, builds an optimized APK, installs it,
and opens it. Use `--device SERIAL` if several devices are connected. Custom signing
keys can be supplied through `CARGO_APK_RELEASE_KEYSTORE` and
`CARGO_APK_RELEASE_KEYSTORE_PASSWORD`.

The package is `org.grafo.launcher_benchmark`; it does not replace Momo. On Android
TV, use the remote's left/right keys for pages and its center button for settings.
The same controls are available by touch or mouse.

## Controls

| Key | Action |
| --- | --- |
| A | Automatic sequence of all workloads |
| P | Repeat page switches |
| S | Repeat settings open/close |
| C | Repeat settings scrolling |
| O | Repeat power-menu open/close |
| Up / Down | Scroll settings manually |
| Left / Right | Switch one page manually |
| Enter | Open or close settings manually |
| Space | Pause/resume the automatic sequence |
| V | Toggle vsync and restart measurements |
| D | Toggle 3/18 apps per page and restart measurements |
| R | Reset results and warm up again |
| F | Toggle fullscreen |
| Escape / Back | Return from power menu, close settings, or exit |

The header has Auto, Pages, Settings, Scroll, Power, Pause, and Load buttons. Tap
the clock/settings pill to open or close the panel, or swipe horizontally to change
pages. Use the wheel or swipe vertically over the settings tiles to scroll. Tap
the power icon to enter the power menu and Back to return. Tap the right half of
the metrics panel to toggle vsync.

## Reading the results

- The first complete automatic cycle is warmup. Select A, P, S, C, or O after manual
  interaction to start a fresh comparable run.
- FPS counts successful render calls over active animation time, including event
  delivery and failed surface acquisitions. Idle holds, warmup, and paused time
  are excluded.
- Work mean, p95, and max measure scene updates plus successful render calls. They
  include blocking inside the renderer, the first draw of each transition, and the
  final draw that applies the settled pose and removes an outgoing menu.
  The p95 uses fixed 0.5 ms histogram buckets. Above 255.5 ms it reports the maximum
  sample instead of understating a large stall.
- Update and render are wall-clock measurements, not GPU timestamps. GPU execution
  can overlap CPU work, and presentation/acquisition can block. This example does
  not enable Grafo's `render_metrics` feature because it forces a GPU wait after
  each frame. Do not enable that feature when comparing normal throughput.
- Vsync is off by default through Grafo's `AutoNoVsync` presentation setting.
  The driver may select mailbox or fall back to FIFO, and the window compositor
  can still limit delivery. Report the displayed resolution, vsync request, and
  grid density with results. FPS here is throughput, not measured display FPS.
- Labels use Grafo's built-in CosmicText adapter with buffers shaped on view entry.
  Moving labels reuse those buffers and Grafo's glyph cache. Metrics use reusable
  text buffers and a fixed histogram. The overlay text is reshaped between
  transitions so that work does not interrupt measured animations.

Use the same scenario, density, resolution, and vsync setting on both revisions.
Let several measured cycles run and watch for thermal throttling on the G31.
Summary lines are emitted every two seconds to the desktop console or Android logcat:

```sh
adb logcat -s grafo-benchmark
```

Each line includes the workload, sample count, FPS, mean, p95 and maximum work time,
resolution, grid density, and vsync request.

## Font

Inter is embedded in the executable and requires no installed system fonts or
network access. See [assets/NOTICE.md](assets/NOTICE.md) and
[the bundled SIL Open Font License](assets/OFL.txt). Include those notices if
redistributing the example or its binaries.
