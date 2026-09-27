use super::{RenderBackend, Renderer};

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    /// Reads tightly packed BGRA pixels, leaving `buffer` unchanged on readback failure.
    pub fn render_to_buffer(&mut self, buffer: &mut Vec<u8>) -> Result<(), B::Error> {
        self.render_with(|backend, commands, _| backend.render_to_buffer(commands, buffer))
    }

    /// Reads ARGB pixels into the viewport-sized prefix, leaving it unchanged on failure.
    pub fn render_to_argb32(&mut self, out_pixels: &mut [u32]) -> Result<(), B::Error> {
        self.render_with(|backend, commands, _| backend.render_to_argb32(commands, out_pixels))
    }
}
