use super::{ClipBackdropBase, ClipBackdropComposition, ClipBackdropFilter, ClipCapture};
use crate::commands::{CaptureConfig, CaptureSource};
use crate::core::PhysicalRect;
use crate::wgpu_backend::execution::backdrops::{self, CapturePurpose};
use crate::wgpu_backend::execution::effects::PooledTexture;
use crate::wgpu_backend::execution::instructions::ExecutionResources;
use crate::wgpu_backend::execution::shapes::coverage::clipping::ClipCoverage;
use crate::wgpu_backend::types::BackdropSource;
use std::sync::Arc;
use wgpu::{CommandEncoder, Texture, TextureView};

pub(in crate::wgpu_backend::execution) struct ClipBackdropSources {
    pub uses_physical_input: bool,
    pub working: ClipCapture,
    pub coverage: Option<Arc<ClipCoverage>>,
    pub inherited: Option<ClipBackdropBase>,
    pub foreground: TextureView,
    pub foreground_bounds: PhysicalRect,
    pub physical_base: Texture,
    pub physical_base_view: TextureView,
    pub has_local_layer: bool,
}

impl ExecutionResources<'_> {
    fn save_clip_backdrop(
        &mut self,
        encoder: &mut CommandEncoder,
        capture: CaptureConfig,
        source: BackdropSource<'_>,
    ) -> usize {
        let texture = backdrops::capture_backdrop(
            encoder,
            &self.context.backdrops,
            source,
            capture.source_rect,
            capture.destination_bounds,
            self.effects,
            &mut self.textures.pool,
        );
        let index = self.textures.work_textures.len();
        self.textures.work_textures.push(texture);
        index
    }

    fn capture_inherited_backdrop(
        &mut self,
        encoder: &mut CommandEncoder,
        capture: CaptureConfig,
        sources: &ClipBackdropSources,
        conditional_base: &Texture,
        conditional_base_view: &TextureView,
        conditional: usize,
    ) -> Option<PooledTexture> {
        let inherited = sources.inherited.as_ref()?;
        let prefix = if sources.has_local_layer {
            self.save_clip_backdrop(
                encoder,
                capture,
                BackdropSource::Layered {
                    base_texture: conditional_base,
                    base_origin: inherited.bounds.min.to_array(),
                    foreground_view: &sources.foreground,
                    foreground_bounds: sources.foreground_bounds,
                },
            )
        } else {
            conditional
        };
        let output = backdrops::capture_backdrop(
            encoder,
            &self.context.backdrops,
            BackdropSource::Flattened {
                texture: &sources.physical_base,
                origin: inherited.bounds.min.to_array(),
            },
            capture.source_rect,
            capture.destination_bounds,
            self.effects,
            &mut self.textures.pool,
        );
        let allocation = self.shapes.clip_coverage().prepare(
            &inherited.coverage,
            None,
            None,
            self.context.device,
            self.context.queue,
            &self.context.pipelines.shapes,
        );
        let composition = ClipBackdropComposition {
            bounds: capture.destination_bounds,
            base_bounds: inherited.bounds,
            coverage: allocation,
            images: [
                conditional_base_view.clone(),
                self.textures.work_textures[prefix].color_view.clone(),
                sources.physical_base_view.clone(),
            ],
        };
        self.compose_inherited_backdrop(encoder, &output, composition);
        Some(output)
    }

    /// Retains both colors across a content target without revisiting its ancestors.
    pub(in crate::wgpu_backend::execution) fn prepare_clip_capture(
        &mut self,
        encoder: &mut CommandEncoder,
        capture: CaptureConfig,
        purpose: CapturePurpose,
        sources: ClipBackdropSources,
    ) -> Option<PooledTexture> {
        let coverage = match (&sources.inherited, &sources.coverage) {
            (Some(base), Some(coverage)) => self
                .shapes
                .clip_coverage()
                .intersection(&base.coverage, coverage),
            (Some(base), None) => Arc::clone(&base.coverage),
            (None, Some(coverage)) => Arc::clone(coverage),
            (None, None) => return None,
        };
        let (conditional_base, conditional_base_view) = match &sources.inherited {
            Some(base) => {
                let texture = &self.textures.work_textures[base.conditional_background];
                (texture.color_texture.clone(), texture.color_view.clone())
            }
            None => (
                sources.physical_base.clone(),
                sources.physical_base_view.clone(),
            ),
        };
        let source = match capture.source {
            CaptureSource::Target => BackdropSource::Flattened {
                texture: &sources.working.texture,
                origin: sources.working.bounds.min.to_array(),
            },
            CaptureSource::Layered { origin, .. } => BackdropSource::Layered {
                base_texture: &conditional_base,
                base_origin: origin,
                foreground_view: &sources.working.view,
                foreground_bounds: sources.working.bounds,
            },
        };
        let conditional = self.save_clip_backdrop(encoder, capture, source);
        match purpose {
            CapturePurpose::BackdropFilter => {
                let filter_coverage = if sources.uses_physical_input {
                    sources.coverage.as_ref()
                } else {
                    Some(&coverage)
                };
                let Some(filter_coverage) = filter_coverage else {
                    return self.capture_inherited_backdrop(
                        encoder,
                        capture,
                        &sources,
                        &conditional_base,
                        &conditional_base_view,
                        conditional,
                    );
                };
                let working = if sources.uses_physical_input && sources.inherited.is_some() {
                    let origin = match capture.source {
                        CaptureSource::Layered { origin, .. } => origin,
                        CaptureSource::Target => sources.working.bounds.min.to_array(),
                    };
                    let source = BackdropSource::Layered {
                        base_texture: &sources.physical_base,
                        base_origin: origin,
                        foreground_view: &sources.working.view,
                        foreground_bounds: sources.working.bounds,
                    };
                    self.save_clip_backdrop(encoder, capture, source)
                } else {
                    conditional
                };
                let allocation = self.shapes.clip_coverage().prepare(
                    filter_coverage,
                    None,
                    None,
                    self.context.device,
                    self.context.queue,
                    &self.context.pipelines.shapes,
                );
                self.textures.clip_filter = Some(ClipBackdropFilter {
                    input: capture.output,
                    bounds: capture.destination_bounds,
                    coverage: allocation,
                    working: self.textures.work_textures[working].color_view.clone(),
                });
            }
            CapturePurpose::BackdropBase => {
                self.textures.clip_backdrop_bases.insert(
                    capture.output,
                    ClipBackdropBase {
                        conditional_background: conditional,
                        bounds: capture.destination_bounds,
                        coverage,
                    },
                );
            }
            _ => {
                unreachable!("clip capture context belongs to a backdrop filter or content target")
            }
        }
        self.capture_inherited_backdrop(
            encoder,
            capture,
            &sources,
            &conditional_base,
            &conditional_base_view,
            conditional,
        )
    }
}
