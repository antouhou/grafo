use super::{RenderBackend, Renderer};

impl<'surface, B: RenderBackend<'surface>> Renderer<'surface, B> {
    pub fn print_memory_usage_info(&self) {
        self.scene.print_memory_usage_info();
        self.backend.print_memory_usage_info();
    }
}
