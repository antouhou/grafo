use super::types::DrawCommandError;
use super::{RenderBackend, Renderer};
use crate::commands::{RenderCommand, RenderOperation, RenderPlan, ShapeDrawId, Target};
use crate::core::{
    CachedShapeHandle, Color, Shape, ShapeDrawCommandOptions, ShapeEffectConfig, ShapeInstance,
    Stroke, Viewport,
};
use crate::render_backend::TextureManager;
use crate::scene::SceneContext;
use thiserror::Error;

#[derive(Default)]
struct TestSurface {
    draws: Vec<usize>,
    shape_masks: usize,
    effects: Vec<u64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("backend unavailable")]
struct TestBackendError;

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
    registered_shapes: Vec<usize>,
    command_address: usize,
    instruction_address: usize,
    should_fail: bool,
}

impl RenderBackend<'_> for TestBackend {
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
            physical_size: (32, 32),
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
    fn resize(&mut self, _: &mut TestSurface, _: Viewport, _: f32) {}
    fn set_msaa_samples(&mut self, _: u32) {}
    fn configure_surface(&mut self, _: &mut TestSurface) {}
    fn set_vsync(&mut self, _: &mut TestSurface, _: bool) {}
    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: &mut TestSurface,
    ) -> Result<(), Self::Error> {
        if self.should_fail {
            return Err(TestBackendError);
        }
        self.command_address = commands as *const RenderPlan as usize;
        self.instruction_address = commands.instructions.as_ptr() as usize;
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
                        RenderOperation::CompositeTexture(_) => continue,
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

    fn render_to_buffer(
        &mut self,
        _: &RenderPlan,
        _: &mut Vec<u8>,
    ) -> Result<(), TestBackendError> {
        Err(TestBackendError)
    }

    fn render_to_argb32(&mut self, _: &RenderPlan, _: &mut [u32]) -> Result<(), TestBackendError> {
        Err(TestBackendError)
    }
}

fn queue_shape(renderer: &mut Renderer<'static, TestBackend>, with_effects: bool) -> usize {
    renderer.load_shape(
        Shape::rect([(0.0, 0.0), (16.0, 16.0)], Stroke::default()),
        1,
        Some(1),
    );
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

fn renderer() -> Renderer<'static, TestBackend> {
    Renderer::from_backend(
        TestBackend::default(),
        TestSurface::default(),
        SceneContext::default(),
    )
}

#[test]
fn render_submits_planned_shapes_and_effects() {
    let mut renderer = renderer();
    let shape = queue_shape(&mut renderer, true);
    let planned_address = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
    ) as *const RenderPlan as usize;
    renderer.render().unwrap();
    assert_eq!(renderer.backend.command_address, planned_address);
    assert_eq!(renderer.surface.draws, [shape]);
    assert_eq!(renderer.surface.shape_masks, 1);
    assert_eq!(renderer.surface.effects, [7, 8]);
}

#[test]
fn effect_parameter_updates_reuse_storage() {
    let mut renderer = renderer();
    let shape = queue_shape(&mut renderer, true);
    renderer.render().unwrap();
    let parameters = renderer.scene.shape_effect(shape).unwrap().parameters;
    renderer
        .update_shape_effect_params(shape, &[1, 2, 3, 4])
        .unwrap();
    renderer
        .update_group_effect_params(shape, &[1, 2, 3, 4])
        .unwrap();
    renderer.render().unwrap();
    let plan = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
    );
    assert_eq!(plan.effect_parameters.len(), 8);
    assert_eq!(plan.parameters(parameters), &[1, 2, 3, 4]);
}

#[test]
fn queue_rebuilds_reuse_command_storage() {
    let mut renderer = renderer();
    let shape = queue_shape(&mut renderer, true);
    renderer.render().unwrap();
    let command_address = renderer.backend.command_address;
    let instruction_address = renderer.backend.instruction_address;
    let parameters = renderer.scene.shape_effect(shape).unwrap().parameters;

    for _ in 0..3 {
        renderer.clear_draw_queue();
        let rebuilt = queue_shape(&mut renderer, true);
        assert_eq!(rebuilt, shape);
        renderer.render().unwrap();
        assert_eq!(renderer.backend.command_address, command_address);
        assert_eq!(renderer.backend.instruction_address, instruction_address);
        assert_eq!(renderer.surface.effects, [7, 8]);
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
    queue_shape(&mut renderer, true);
    renderer.render().unwrap();

    renderer.clear_draw_queue();
    renderer.render().unwrap();
    assert!(renderer.surface.draws.is_empty());
    assert!(renderer.surface.effects.is_empty());
    assert_eq!(renderer.surface.shape_masks, 0);
    let plan = renderer.planner.plan(
        &renderer.scene,
        renderer.viewport,
        renderer.fringe_width,
        4096,
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
    let shape = queue_shape(&mut first, false);
    let mut second = renderer();
    queue_shape(&mut second, false);
    let commands = first
        .planner
        .plan(&first.scene, first.viewport, first.fringe_width, 4096);
    first.backend.render(commands, &mut first.surface).unwrap();
    second
        .backend
        .render(commands, &mut second.surface)
        .unwrap();
    assert_eq!(first.surface.draws, [shape]);
    assert_eq!(second.surface.draws, [shape]);
    assert_eq!(
        first.backend.command_address,
        second.backend.command_address
    );

    second.clear_draw_queue();
    second.render().unwrap();
    assert!(second.surface.draws.is_empty());
    assert_eq!(first.surface.draws, [shape]);
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
    let shape = queue_shape(&mut renderer, false);
    renderer.render().unwrap();

    renderer.backend.should_fail = true;
    assert_eq!(renderer.render(), Err(TestBackendError));
    assert_eq!(renderer.surface.draws, [shape]);

    renderer.backend.should_fail = false;
    renderer.clear_draw_queue();
    renderer.render().unwrap();
    assert!(renderer.surface.draws.is_empty());
}
