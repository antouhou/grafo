use super::{DrawPlanner, DrawPlanningInput, DrawTargetState};
use crate::commands::{CaptureConfig, CaptureSource, RenderOperation};
use crate::core::effect::CaptureRegion;
use crate::planner::commands::TraversalCommands;
use crate::planner::content_effects::ContentCapture;
use crate::planner::TextureIdAllocator;

impl DrawPlanner {
    pub(super) fn begin_content_capture<'scene>(
        &mut self,
        node_id: usize,
        input: &DrawPlanningInput<'scene, '_>,
        output: &mut TraversalCommands<'scene>,
        texture_ids: &mut TextureIdAllocator,
    ) -> Option<ContentCapture> {
        let effect = input
            .content_effects
            .get(&node_id)
            .filter(|effect| effect.bounds.is_some())?;
        let source = self.current.backdrop_source;
        let backdrop_region = self.subtree_backdrop_regions.get(&node_id).copied();
        let backdrop_source = match backdrop_region {
            Some(region) => {
                let base = texture_ids.allocate();
                output.push(RenderOperation::Capture(CaptureConfig {
                    source,
                    region: CaptureRegion {
                        bounds: region.to_i32(),
                        source_rect: Some(region),
                    },
                    output: base,
                    sampling_size: region.size(),
                }));
                CaptureSource::Layered {
                    base,
                    origin: region.min.to_i32().to_array(),
                }
            }
            None => CaptureSource::Target,
        };
        let capture = ContentCapture::new(*effect, backdrop_region, texture_ids)?;
        capture.append_begin(output);
        self.current = DrawTargetState::for_viewport(input.physical_size, backdrop_source);
        Some(capture)
    }
}
