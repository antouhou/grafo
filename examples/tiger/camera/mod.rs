use grafo::TransformInstance;

/// Maps artwork coordinates to logical window coordinates.
pub struct Camera {
    scale: f64,
    offset: (f64, f64),
    viewport: (f64, f64),
}

impl Camera {
    pub fn new(bounds: (f64, f64, f64, f64), viewport: (f64, f64)) -> Self {
        let scale = ((viewport.0 - 64.0) / bounds.2)
            .min((viewport.1 - 64.0) / bounds.3)
            .clamp(0.02, 64.0);
        Self {
            scale,
            offset: (
                (viewport.0 - bounds.2 * scale) * 0.5 - bounds.0 * scale,
                (viewport.1 - bounds.3 * scale) * 0.5 - bounds.1 * scale,
            ),
            viewport,
        }
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn transform(&self) -> TransformInstance {
        TransformInstance::scale(self.scale as f32, self.scale as f32).then(
            &TransformInstance::translation(self.offset.0 as f32, self.offset.1 as f32),
        )
    }

    pub fn pan(&mut self, delta: (f64, f64)) {
        self.offset.0 += delta.0;
        self.offset.1 += delta.1;
    }

    /// Keeps the artwork point beneath the cursor fixed, including at zoom limits.
    pub fn zoom(&mut self, cursor: (f64, f64), scroll_lines: f64) {
        let scale = (self.scale * (scroll_lines * 0.15).exp()).clamp(0.02, 64.0);
        let ratio = scale / self.scale;
        self.offset.0 = cursor.0 - (cursor.0 - self.offset.0) * ratio;
        self.offset.1 = cursor.1 - (cursor.1 - self.offset.1) * ratio;
        self.scale = scale;
    }

    pub fn resize(&mut self, viewport: (f64, f64)) {
        self.pan((
            (viewport.0 - self.viewport.0) * 0.5,
            (viewport.1 - self.viewport.1) * 0.5,
        ));
        self.viewport = viewport;
    }
}

#[cfg(test)]
mod tests;
