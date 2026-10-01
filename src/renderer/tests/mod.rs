use super::types::DrawCommandError;
use super::{RenderBackend, Renderer};
use crate::commands::{RenderCommand, RenderOperation, RenderPlan, ShapeDrawId, Target};
use crate::core::{
    CachedShapeHandle, Color, Shape, ShapeDrawCommandOptions, ShapeEffectConfig, ShapeInstance,
    UnsignedPhysicalRect, Viewport,
};
use crate::render_backend::render_target::{
    PixelFormat, PixelLayout, Pixmap, PixmapMut, RenderTarget, RenderTargetError, Surface,
};
use crate::render_backend::TextureManager;
use crate::scene::SceneContext;
use thiserror::Error;

mod dirty_bounds;
mod removal;
mod shape_effect_damage;

#[derive(Default)]
struct TestSurface {
    draws: Vec<usize>,
    shape_masks: usize,
    effects: Vec<u64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("backend unavailable")]
struct TestBackendError;

impl From<RenderTargetError> for TestBackendError {
    fn from(_: RenderTargetError) -> Self {
        Self
    }
}

struct TestTextureManager;

impl TextureManager for TestTextureManager {
    type Error = TestBackendError;

    fn clear(&self) {
        unreachable!("this test backend does not use textures");
    }

    fn allocate_texture(&self, _: u64, _: (u32, u32)) {
        unreachable!("this test backend does not use textures");
    }

    fn allocate_texture_with_data(&self, _: u64, _: (u32, u32), _: &[u8]) {
        unreachable!("this test backend does not use textures");
    }

    fn load_data_into_texture(&self, _: u64, _: (u32, u32), _: &[u8]) -> Result<(), Self::Error> {
        unreachable!("this test backend does not use textures");
    }

    fn remove_texture(&self, _: u64) {
        unreachable!("this test backend does not use textures");
    }

    fn is_texture_loaded(&self, _: u64) -> bool {
        unreachable!("this test backend does not use textures");
    }
}

#[derive(Default)]
struct TestBackend {
    root_scissor: Option<UnsignedPhysicalRect>,
    is_dirty_region_overlay_enabled: bool,
    registered_shapes: Vec<usize>,
    command_address: usize,
    instruction_address: usize,
    should_fail: bool,
    size: Option<(u32, u32)>,
}

impl RenderBackend for TestBackend {
    type Surface = TestSurface;
    type Error = TestBackendError;
    type TextureManager = TestTextureManager;
    fn register_shape(
        &mut self,
        id: ShapeDrawId,
        shape: &ShapeInstance,
    ) -> Result<(), TestBackendError> {
        if self.should_fail {
            return Err(TestBackendError);
        }
        assert!(!shape.cached_shape.vertex_buffers().vertices.is_empty());
        self.registered_shapes.push(id.0);
        Ok(())
    }

    fn unregister_shapes(&mut self, ids: &[ShapeDrawId]) {
        self.registered_shapes
            .retain(|&registered| !ids.contains(&ShapeDrawId(registered)));
    }

    fn clear_draw_queue(&mut self) {
        self.registered_shapes.clear();
    }

    fn texture_manager(&self) -> &TestTextureManager {
        &TestTextureManager
    }

    fn maximum_texture_dimension(&self) -> u32 {
        4096
    }

    fn viewport(&self) -> Viewport {
        Viewport {
            physical_size: self.size.unwrap_or((32, 32)),
            scale_factor: 1.0,
        }
    }

    fn fringe_width(&self) -> f32 {
        0.75
    }

    fn load_effect(&mut self, _: u64, _: &[&str]) -> Result<bool, TestBackendError> {
        Ok(true)
    }

    fn validate_effect_params(&self, _: u64, params: &[u8]) -> Result<(), TestBackendError> {
        if params.len() == 4 {
            Ok(())
        } else {
            Err(TestBackendError)
        }
    }

    fn unload_effect(&mut self, _: u64) {}
    fn invalidate_effect(&mut self, _: u64) {}
    fn set_shape_effect_geometry(&mut self, id: ShapeDrawId, _: &CachedShapeHandle) {
        assert!(self.registered_shapes.contains(&id.0));
    }

    fn remove_shape_effect(&mut self, _: ShapeDrawId) {}
    fn remove_backdrop_effect(&mut self, _: ShapeDrawId) {}
    fn resize(&mut self, viewport: Viewport, _: f32) {
        self.size = Some(viewport.physical_size);
    }
    fn set_msaa_samples(&mut self, _: u32) {}
    fn set_dirty_region_overlay_enabled(&mut self, enabled: bool) {
        self.is_dirty_region_overlay_enabled = enabled;
    }

    fn is_dirty_region_overlay_enabled(&self) -> bool {
        self.is_dirty_region_overlay_enabled
    }

    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: RenderTarget<'_, TestSurface>,
    ) -> Result<(), Self::Error> {
        if self.should_fail {
            return Err(TestBackendError);
        }
        self.command_address = commands as *const RenderPlan as usize;
        self.root_scissor = commands.root_scissor;
        self.instruction_address = commands.instructions.as_ptr() as usize;
        let surface = match surface {
            RenderTarget::Surface(surface) => surface.resource_mut(),
            RenderTarget::Pixmap(mut pixels) => {
                let layout = pixels.layout();
                for row in 0..layout.size().1 as usize {
                    let start = row * layout.stride();
                    pixels.pixels_mut()[start..start + layout.size().0 as usize * 4].fill(71);
                }
                return Ok(());
            }
        };
        surface.draws.clear();
        surface.effects.clear();
        surface.shape_masks = 0;
        let mut targets = Vec::new();
        for command in &commands.instructions {
            match &command.operation {
                RenderOperation::BeginTarget(target) => targets.push(*target),
                RenderOperation::EndTarget => {
                    targets.pop().expect("balanced target scopes");
                }
                RenderOperation::DrawShapeMask(mask) => {
                    assert!(matches!(targets.last(), Some(Target::Mask(_))));
                    assert!(self.registered_shapes.contains(&mask.shape.0));
                    surface.shape_masks += 1;
                }
                RenderOperation::ApplyEffect(effect) => {
                    assert_eq!(commands.parameters(effect.parameters), &[1, 2, 3, 4]);
                    surface.effects.push(effect.effect_id);
                }
                operation => {
                    assert!(!targets.is_empty());
                    let shape = match operation {
                        RenderOperation::DrawShape(draw)
                        | RenderOperation::DrawShapeAndIncrementStencil(draw)
                        | RenderOperation::DecrementStencil(draw) => draw.id,
                        RenderOperation::IncrementStencil(shape) => *shape,
                        RenderOperation::CompositeTexture(_)
                        | RenderOperation::CaptureBackdrop(_) => continue,
                        _ => panic!("unexpected operation"),
                    };
                    assert!(self.registered_shapes.contains(&shape.0));
                    surface.draws.push(shape.0);
                }
            }
        }
        assert!(targets.is_empty());
        Ok(())
    }
}

fn queue_shape(renderer: &mut Renderer<TestBackend>, with_effects: bool) -> usize {
    renderer.load_shape(Shape::rect([(0.0, 0.0), (16.0, 16.0)]), 1, Some(1));
    let id = renderer
        .add_cached_shape(
            1,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    if with_effects {
        renderer.load_effect(7, &["shape effect"]).unwrap();
        renderer.load_effect(8, &["group effect"]).unwrap();
        renderer.set_group_effect(id, 8, &[1, 2, 3, 4]).unwrap();
        renderer
            .set_shape_effect(id, 7, &[1, 2, 3, 4], ShapeEffectConfig::default())
            .unwrap();
    }
    id
}

fn renderer() -> Renderer<TestBackend> {
    Renderer::from_backend(TestBackend::default(), SceneContext::default())
}

fn surface() -> Surface<TestSurface> {
    Surface::from_resource(TestSurface::default(), (32, 32), true)
}

#[test]
fn render_submits_planned_shapes_and_effects() {
    let mut renderer = renderer();
    let mut surface = surface();
    let shape = queue_shape(&mut renderer, true);
    let planned_address = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
        None,
    ) as *const RenderPlan as usize;
    renderer.render(&mut surface).unwrap();
    assert_eq!(renderer.backend.command_address, planned_address);
    assert_eq!(surface.resource().draws, [shape]);
    assert_eq!(surface.resource().shape_masks, 1);
    assert_eq!(surface.resource().effects, [7, 8]);
}

#[test]
fn effect_parameter_updates_reuse_storage() {
    let mut renderer = renderer();
    let mut surface = surface();
    let shape = queue_shape(&mut renderer, true);
    renderer.render(&mut surface).unwrap();
    let parameters = renderer.scene.shape_effect(shape).unwrap().parameters;
    renderer
        .update_shape_effect_params(shape, &[1, 2, 3, 4])
        .unwrap();
    renderer
        .update_group_effect_params(shape, &[1, 2, 3, 4])
        .unwrap();
    renderer.render(&mut surface).unwrap();
    let plan = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
        None,
    );
    assert_eq!(plan.effect_parameters.len(), 8);
    assert_eq!(plan.parameters(parameters), &[1, 2, 3, 4]);
}

#[test]
fn queue_rebuilds_reuse_command_storage() {
    let mut renderer = renderer();
    let mut surface = surface();
    let shape = queue_shape(&mut renderer, true);
    renderer.render(&mut surface).unwrap();
    let command_address = renderer.backend.command_address;
    let instruction_address = renderer.backend.instruction_address;
    let parameters = renderer.scene.shape_effect(shape).unwrap().parameters;

    for _ in 0..3 {
        renderer.clear_draw_queue();
        let rebuilt = queue_shape(&mut renderer, true);
        assert_eq!(rebuilt, shape);
        renderer.render(&mut surface).unwrap();
        assert_eq!(renderer.backend.command_address, command_address);
        assert_eq!(renderer.backend.instruction_address, instruction_address);
        assert_eq!(surface.resource().effects, [7, 8]);
        assert_eq!(
            renderer
                .scene
                .shape_effect(rebuilt)
                .unwrap()
                .parameters
                .hash,
            parameters.hash
        );
    }
}

#[test]
fn clearing_queue_removes_planned_draws_and_effects() {
    let mut renderer = renderer();
    let mut surface = surface();
    queue_shape(&mut renderer, true);
    renderer.render(&mut surface).unwrap();

    renderer.clear_draw_queue();
    renderer.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
    assert!(surface.resource().effects.is_empty());
    assert_eq!(surface.resource().shape_masks, 0);
    let plan = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
        None,
    );
    assert!(matches!(
        plan.instructions.as_slice(),
        [
            RenderCommand {
                operation: RenderOperation::BeginTarget(Target::Surface),
                ..
            },
            RenderCommand {
                operation: RenderOperation::EndTarget,
                ..
            }
        ]
    ));
}

#[test]
fn rendering_to_one_surface_does_not_change_another() {
    let mut first = renderer();
    let mut first_surface = surface();
    let shape = queue_shape(&mut first, false);
    let mut second = renderer();
    let mut second_surface = surface();
    queue_shape(&mut second, false);
    let commands = first
        .planner
        .plan(&first.scene, first.viewport, first.fringe_width, 4096, None);
    first
        .backend
        .render(commands, (&mut first_surface).into())
        .unwrap();
    second
        .backend
        .render(commands, (&mut second_surface).into())
        .unwrap();
    assert_eq!(first_surface.resource().draws, [shape]);
    assert_eq!(second_surface.resource().draws, [shape]);
    assert_eq!(
        first.backend.command_address,
        second.backend.command_address
    );

    second.clear_draw_queue();
    second.render(&mut second_surface).unwrap();
    assert!(second_surface.resource().draws.is_empty());
    assert_eq!(first_surface.resource().draws, [shape]);
}

#[test]
fn failed_registration_leaves_scene_unchanged() {
    let mut renderer = renderer();
    let shape = queue_shape(&mut renderer, false);

    renderer.backend.should_fail = true;
    assert!(matches!(
        renderer.add_cached_shape(1, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Backend(TestBackendError))
    ));
    assert!(renderer.scene.shape(shape + 1).is_err());
    assert_eq!(renderer.backend.registered_shapes, [shape]);

    renderer.backend.should_fail = false;
    assert_eq!(
        renderer
            .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
            .unwrap(),
        shape + 1
    );
}

#[test]
fn failed_render_preserves_surface_contents() {
    let mut renderer = renderer();
    let mut surface = surface();
    let shape = queue_shape(&mut renderer, false);
    renderer.render(&mut surface).unwrap();

    renderer.backend.should_fail = true;
    assert_eq!(renderer.render(&mut surface), Err(TestBackendError));
    assert_eq!(surface.resource().draws, [shape]);

    renderer.backend.should_fail = false;
    renderer.clear_draw_queue();
    renderer.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
}

#[test]
fn shared_render_target_contract_accepts_owned_and_borrowed_memory_without_wgpu() {
    let mut renderer = renderer();
    queue_shape(&mut renderer, false);
    let mut owned = Pixmap::new((12, 7), PixelFormat::Bgra8).unwrap();
    renderer.render(&mut owned).unwrap();
    assert!(owned.pixels().iter().all(|byte| *byte == 71));
    assert_eq!(renderer.size(), (12, 7));
    let mut borrowed = [99; 32];
    let layout = PixelLayout::new((2, 2), PixelFormat::Argb32, 12).unwrap();
    renderer
        .render(PixmapMut::new(&mut borrowed, layout).unwrap())
        .unwrap();
    assert_eq!(&borrowed[..8], &[71; 8]);
    assert_eq!(&borrowed[8..12], &[99; 4]);
    assert_eq!(&borrowed[12..20], &[71; 8]);
    assert_eq!(&borrowed[20..], &[99; 12]);
    let before_failure = borrowed;
    renderer.backend.should_fail = true;
    assert_eq!(
        renderer.render(PixmapMut::new(&mut borrowed, layout).unwrap()),
        Err(TestBackendError)
    );
    assert_eq!(borrowed, before_failure);
}

#[test]
fn invalid_surface_size_is_rejected_before_submission() {
    let mut renderer = renderer();
    let mut surface = surface();
    renderer.render(&mut surface).unwrap();
    let previous_size = renderer.size();
    for size in [(0, 32), (4097, 32)] {
        surface.resize(size);
        assert_eq!(renderer.render(&mut surface), Err(TestBackendError));
        assert_eq!(renderer.size(), previous_size);
    }
}
