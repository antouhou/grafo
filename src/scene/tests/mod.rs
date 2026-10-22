use super::{DrawCommandError, RenderBackend, Scene};
use crate::commands::{RenderCommand, RenderOperation, ShapeDrawId, Target};
use crate::core::{
    CachedShapeHandle, Color, ContentEffectConfig, Shape, ShapeDrawCommandOptions, ShapeInstance,
    ShapeMaskEffectConfig, UnsignedPhysicalRect, Viewport,
};
use crate::render_backend::render_target::{
    PixelFormat, PixelLayout, Pixmap, PixmapMut, RenderTarget, RenderTargetError, Surface,
};
use crate::render_backend::texture_manager::{TextureManager, TextureUpdatePublisher};
use crate::scene_state::SceneStateContext;
use ahash::HashSet;
use std::sync::{Arc, Mutex};
use thiserror::Error;

mod damage;
mod removal;
mod replacement;
mod text;

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

#[derive(Clone, Default)]
struct TestTextureManager {
    textures: Arc<Mutex<HashSet<u64>>>,
    updates: TextureUpdatePublisher,
}

impl TextureManager for TestTextureManager {
    type Error = TestBackendError;

    fn clear(&self) {
        let mut textures = self.textures.lock().unwrap();
        for texture_id in textures.drain() {
            self.updates.publish(texture_id);
        }
    }

    fn allocate_texture(&self, texture_id: u64, _: (u32, u32)) {
        self.textures.lock().unwrap().insert(texture_id);
        self.updates.publish(texture_id);
    }

    fn allocate_texture_with_data(&self, texture_id: u64, dimensions: (u32, u32), _: &[u8]) {
        self.allocate_texture(texture_id, dimensions);
    }

    fn load_data_into_texture(
        &self,
        texture_id: u64,
        _: (u32, u32),
        _: &[u8],
    ) -> Result<(), Self::Error> {
        if !self.is_texture_loaded(texture_id) {
            return Err(TestBackendError);
        }
        self.updates.publish(texture_id);
        Ok(())
    }

    fn remove_texture(&self, texture_id: u64) {
        if self.textures.lock().unwrap().remove(&texture_id) {
            self.updates.publish(texture_id);
        }
    }

    fn is_texture_loaded(&self, texture_id: u64) -> bool {
        self.textures.lock().unwrap().contains(&texture_id)
    }
}

#[derive(Default)]
struct TestBackend {
    texture_manager: TestTextureManager,
    root_scissor: Option<UnsignedPhysicalRect>,
    is_dirty_region_overlay_enabled: bool,
    registered_shapes: Vec<usize>,
    refreshed_uvs: Vec<usize>,
    should_fail: bool,
    registration_failure: Option<usize>,
    size: Option<(u32, u32)>,
    texture_upload_during_render: Option<u64>,
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
        if self.should_fail || self.registration_failure == Some(id.0) {
            return Err(TestBackendError);
        }
        let geometry = shape.cached_shape.vertex_buffers();
        assert_eq!(geometry.vertices.is_empty(), geometry.indices.is_empty());
        if !self.registered_shapes.contains(&id.0) {
            self.registered_shapes.push(id.0);
        }
        Ok(())
    }

    fn refresh_uv(&mut self, id: ShapeDrawId, shape: &ShapeInstance) {
        assert!(self.registered_shapes.contains(&id.0));
        assert!(shape
            .textures
            .iter()
            .any(|texture| texture.texture_id.is_some()));
        self.refreshed_uvs.push(id.0);
    }

    fn unregister_shapes(&mut self, ids: &[ShapeDrawId]) {
        self.registered_shapes
            .retain(|&registered| !ids.contains(&ShapeDrawId(registered)));
    }

    fn clear(&mut self) {
        self.registered_shapes.clear();
    }

    fn texture_manager(&self) -> &TestTextureManager {
        &self.texture_manager
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
    fn set_shape_mask_effect_geometry(&mut self, id: ShapeDrawId, _: &CachedShapeHandle) {
        assert!(self.registered_shapes.contains(&id.0));
    }

    fn remove_shape_mask_effect(&mut self, _: ShapeDrawId) {}
    fn remove_backdrop_effect(&mut self, _: ShapeDrawId) {}
    fn resize(&mut self, viewport: Viewport) {
        self.size = Some(viewport.physical_size);
    }
    fn set_msaa_samples(&mut self, _: u32) {}
    fn set_dirty_region_overlay_enabled(&mut self, enabled: bool) {
        self.is_dirty_region_overlay_enabled = enabled;
    }

    fn is_dirty_region_overlay_enabled(&self) -> bool {
        self.is_dirty_region_overlay_enabled
    }

    fn render<'scene>(
        &mut self,
        root_scissor: Option<UnsignedPhysicalRect>,
        commands: impl Iterator<Item = RenderCommand<'scene>>,
        surface: RenderTarget<'_, TestSurface>,
    ) -> Result<(), Self::Error> {
        if self.should_fail {
            return Err(TestBackendError);
        }
        if let Some(texture_id) = self.texture_upload_during_render.take() {
            self.texture_manager
                .load_data_into_texture(texture_id, (1, 1), &[255; 4])?;
        }
        self.root_scissor = root_scissor;
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
        for command in commands {
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
                    assert_eq!(effect.parameters.bytes, &[1, 2, 3, 4]);
                    surface.effects.push(effect.effect_id);
                }
                operation => {
                    assert!(!targets.is_empty());
                    let shape = match operation {
                        RenderOperation::DrawShape(draw)
                        | RenderOperation::DrawShapeAndIncrementStencil(draw)
                        | RenderOperation::DecrementStencil(draw) => draw.id,
                        RenderOperation::CompositeTexture(_) | RenderOperation::Capture(_) => {
                            continue
                        }
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

fn queue_shape(scene: &mut Scene<TestBackend>, with_effects: bool) -> usize {
    scene.load_shape(Shape::rect([(0.0, 0.0), (16.0, 16.0)]), 1, Some(1));
    let id = scene
        .add_cached_shape(
            1,
            None,
            ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )
        .unwrap();
    if with_effects {
        scene.load_effect(7, &["shape mask effect"]).unwrap();
        scene.load_effect(8, &["content effect"]).unwrap();
        scene
            .set_content_effect(id, 8, &[1, 2, 3, 4], ContentEffectConfig::default())
            .unwrap();
        scene
            .set_shape_mask_effect(id, 7, &[1, 2, 3, 4], ShapeMaskEffectConfig::default())
            .unwrap();
    }
    id
}

fn scene() -> Scene<TestBackend> {
    let backend = TestBackend::default();
    let context = SceneStateContext::new(backend.texture_manager.updates.clone());
    Scene::from_parts(backend, context)
}

fn surface() -> Surface<TestSurface> {
    Surface::from_resource(TestSurface::default(), (32, 32), true)
}

#[test]
fn render_submits_planned_shapes_and_effects() {
    let mut scene = scene();
    let mut surface = surface();
    let shape = queue_shape(&mut scene, true);
    scene.render(&mut surface).unwrap();
    assert_eq!(surface.resource().draws, [shape]);
    assert_eq!(surface.resource().shape_masks, 1);
    assert_eq!(surface.resource().effects, [7, 8]);
}

#[test]
fn queue_rebuilds_reuse_scene_parameter_storage() {
    let mut scene = scene();
    let mut surface = surface();
    let shape = queue_shape(&mut scene, true);
    scene.render(&mut surface).unwrap();
    let parameters_address = scene.state.effect_parameters.bytes.as_ptr();
    let parameters_capacity = scene.state.effect_parameters.bytes.capacity();
    let parameters = scene.state.shape_mask_effect(shape).unwrap().parameters;

    for _ in 0..3 {
        scene.clear();
        let rebuilt = queue_shape(&mut scene, true);
        assert_eq!(rebuilt, shape);
        scene.render(&mut surface).unwrap();
        assert_eq!(
            scene.state.effect_parameters.bytes.as_ptr(),
            parameters_address
        );
        assert_eq!(
            scene.state.effect_parameters.bytes.capacity(),
            parameters_capacity
        );
        assert_eq!(surface.resource().effects, [7, 8]);
        assert_eq!(
            scene
                .state
                .shape_mask_effect(rebuilt)
                .unwrap()
                .parameters
                .hash,
            parameters.hash
        );
    }
}

#[test]
fn clearing_queue_removes_planned_draws_and_effects() {
    let mut scene = scene();
    let mut surface = surface();
    queue_shape(&mut scene, true);
    scene.render(&mut surface).unwrap();

    scene.clear();
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
    assert!(surface.resource().effects.is_empty());
    assert_eq!(surface.resource().shape_masks, 0);
    let plan: Vec<_> = scene.planner.plan(&scene.state, 4096).collect();
    assert!(matches!(
        plan.as_slice(),
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
    let mut first = scene();
    let mut first_surface = surface();
    let first_shape = queue_shape(&mut first, false);
    let mut second = scene();
    let mut second_surface = surface();
    let second_shapes = [
        queue_shape(&mut second, false),
        queue_shape(&mut second, false),
    ];
    let commands = first.planner.plan(&first.state, 4096);
    first
        .backend
        .render(first.dirty_bounds, commands, (&mut first_surface).into())
        .unwrap();
    let commands = second.planner.plan(&second.state, 4096);
    second
        .backend
        .render(second.dirty_bounds, commands, (&mut second_surface).into())
        .unwrap();
    assert_eq!(first_surface.resource().draws, [first_shape]);
    assert_eq!(second_surface.resource().draws, second_shapes);

    second.clear();
    second.render(&mut second_surface).unwrap();
    assert!(second_surface.resource().draws.is_empty());
    assert_eq!(first_surface.resource().draws, [first_shape]);
}

#[test]
fn failed_registration_leaves_scene_unchanged() {
    let mut scene = scene();
    let shape = queue_shape(&mut scene, false);

    scene.backend.should_fail = true;
    assert!(matches!(
        scene.add_cached_shape(1, None, ShapeDrawCommandOptions::new()),
        Err(DrawCommandError::Backend(TestBackendError))
    ));
    assert!(scene.state.shape(shape + 1).is_err());
    assert_eq!(scene.backend.registered_shapes, [shape]);

    scene.backend.should_fail = false;
    assert_eq!(
        scene
            .add_cached_shape(1, None, ShapeDrawCommandOptions::new())
            .unwrap(),
        shape + 1
    );
}

#[test]
fn failed_render_preserves_surface_contents() {
    let mut scene = scene();
    let mut surface = surface();
    let shape = queue_shape(&mut scene, false);
    scene.render(&mut surface).unwrap();

    scene.backend.should_fail = true;
    assert_eq!(scene.render(&mut surface), Err(TestBackendError));
    assert_eq!(surface.resource().draws, [shape]);

    scene.backend.should_fail = false;
    scene.clear();
    scene.render(&mut surface).unwrap();
    assert!(surface.resource().draws.is_empty());
}

#[test]
fn shared_render_target_contract_accepts_owned_and_borrowed_memory_without_wgpu() {
    let mut scene = scene();
    queue_shape(&mut scene, false);
    let mut owned = Pixmap::new((12, 7), PixelFormat::Bgra8).unwrap();
    scene.render(&mut owned).unwrap();
    assert!(owned.pixels().iter().all(|byte| *byte == 71));
    assert_eq!(scene.size(), (12, 7));
    let mut borrowed = [99; 32];
    let layout = PixelLayout::new((2, 2), PixelFormat::Argb32, 12).unwrap();
    scene
        .render(PixmapMut::new(&mut borrowed, layout).unwrap())
        .unwrap();
    assert_eq!(&borrowed[..8], &[71; 8]);
    assert_eq!(&borrowed[8..12], &[99; 4]);
    assert_eq!(&borrowed[12..20], &[71; 8]);
    assert_eq!(&borrowed[20..], &[99; 12]);
    let before_failure = borrowed;
    scene.backend.should_fail = true;
    assert_eq!(
        scene.render(PixmapMut::new(&mut borrowed, layout).unwrap()),
        Err(TestBackendError)
    );
    assert_eq!(borrowed, before_failure);
}

#[test]
fn invalid_surface_size_is_rejected_before_submission() {
    let mut scene = scene();
    let mut surface = surface();
    scene.render(&mut surface).unwrap();
    let previous_size = scene.size();
    for size in [(0, 32), (4097, 32)] {
        surface.resize(size);
        assert_eq!(scene.render(&mut surface), Err(TestBackendError));
        assert_eq!(scene.size(), previous_size);
    }
}
