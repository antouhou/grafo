/// A single pixel-color expectation to validate after rendering.
pub struct PixelExpectation {
    pub x: u32,
    pub y: u32,
    pub expected_r: u8,
    pub expected_g: u8,
    pub expected_b: u8,
    pub expected_a: u8,
    /// Per-channel tolerance for comparison (default 5).
    pub tolerance: u8,
    /// Human-readable label for failure messages.
    pub label: &'static str,
}

impl PixelExpectation {
    pub fn new(x: u32, y: u32, r: u8, g: u8, b: u8, a: u8, label: &'static str) -> Self {
        Self {
            x,
            y,
            expected_r: r,
            expected_g: g,
            expected_b: b,
            expected_a: a,
            tolerance: 5,
            label,
        }
    }

    pub fn with_tolerance(mut self, tolerance: u8) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Convenience: expect a fully opaque color.
    pub fn opaque(x: u32, y: u32, r: u8, g: u8, b: u8, label: &'static str) -> Self {
        Self::new(x, y, r, g, b, 255, label)
    }

    /// Convenience: expect a fully transparent pixel.
    pub fn transparent(x: u32, y: u32, label: &'static str) -> Self {
        Self::new(x, y, 0, 0, 0, 0, label)
    }
}

/// Validates pixel expectations against raw BGRA8 pixel data from `render_to_buffer()`.
///
/// Returns a list of human-readable failure descriptions. An empty list means
/// all expectations passed.
pub fn check_pixels(
    pixel_data: &[u8],
    width: u32,
    height: u32,
    expectations: &[PixelExpectation],
) -> Vec<String> {
    let mut failures = Vec::new();
    let stride = (width as usize) * 4;

    for expectation in expectations {
        if expectation.x >= width || expectation.y >= height {
            failures.push(format!(
                "[{}] pixel ({},{}) is outside canvas {}×{}",
                expectation.label, expectation.x, expectation.y, width, height,
            ));
            continue;
        }

        let offset = (expectation.y as usize) * stride + (expectation.x as usize) * 4;
        if offset + 4 > pixel_data.len() {
            failures.push(format!(
                "[{}] pixel ({},{}) is out of bounds (buffer len {})",
                expectation.label,
                expectation.x,
                expectation.y,
                pixel_data.len(),
            ));
            continue;
        }

        // render_to_buffer returns BGRA8
        let actual_b = pixel_data[offset];
        let actual_g = pixel_data[offset + 1];
        let actual_r = pixel_data[offset + 2];
        let actual_a = pixel_data[offset + 3];

        let tolerance = expectation.tolerance as i16;
        let matches = channel_matches(actual_r, expectation.expected_r, tolerance)
            && channel_matches(actual_g, expectation.expected_g, tolerance)
            && channel_matches(actual_b, expectation.expected_b, tolerance)
            && channel_matches(actual_a, expectation.expected_a, tolerance);

        if !matches {
            failures.push(format!(
                "[{}] pixel ({},{}) expected rgba({},{},{},{}) ±{} but got rgba({},{},{},{})",
                expectation.label,
                expectation.x,
                expectation.y,
                expectation.expected_r,
                expectation.expected_g,
                expectation.expected_b,
                expectation.expected_a,
                expectation.tolerance,
                actual_r,
                actual_g,
                actual_b,
                actual_a,
            ));
        }
    }

    failures
}

fn channel_matches(actual: u8, expected: u8, tolerance: i16) -> bool {
    let diff = (actual as i16) - (expected as i16);
    diff.abs() <= tolerance
}
