use super::plan::Planner;
use super::types::DrawTreeNode;
use super::{RenderBackend, Renderer, Viewport, DEFAULT_FRINGE_WIDTH};
use crate::commands::{DrawOperation, DrawSegment, RenderPlan, Target};
use crate::core::effect::{EffectInstance, ShapeEffectInstance};
use crate::core::shape::CachedShapeHandle;
use crate::renderer::types::CachedShapeDrawData;
use crate::{Color, Shape, ShapeDrawCommandOptions, ShapeEffectConfig, Stroke};
use ahash::{HashMap, HashMapExt};
use std::mem;
use std::sync::{Arc, RwLock};
#[cfg(feature = "render_metrics")]
use std::time::Duration;
use thiserror::Error;

#[derive(Default)]
struct TestSurface {
    draws: Vec<usize>,
    shape_masks: usize,
    effects: Vec<u64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("surface unavailable")]
struct TestRenderError;

#[derive(Default)]
struct TestBackend {
    registered_shapes: Vec<usize>,
    command_address: usize,
    instruction_address: usize,
    mask_address: usize,
    should_fail: bool,
}

impl RenderBackend<'_> for TestBackend {
    type Surface = TestSurface;
    type Error = TestRenderError;

    fn render(
        &mut self,
        commands: &RenderPlan,
        surface: &mut TestSurface,
    ) -> Result<(), Self::Error> {
        if self.should_fail {
            return Err(TestRenderError);
        }
        self.command_address = commands as *const RenderPlan as usize;
        self.instruction_address = commands.scene.instructions.as_ptr() as usize;
        self.mask_address = commands.shape_effects.segments.as_ptr() as usize;
        surface.draws.clear();
        surface.effects.clear();
        surface.shape_masks = 0;
        for plan in [&commands.shape_effects, &commands.scene] {
            let mut has_target = false;
            for segment in &plan.segments {
                match segment {
                    DrawSegment::BeginTarget(_) => assert!(!mem::replace(&mut has_target, true)),
                    DrawSegment::EndTarget => assert!(mem::replace(&mut has_target, false)),
                    DrawSegment::DrawShapeMask(mask) => {
                        assert!(has_target);
                        assert!(self.registered_shapes.contains(&mask.shape.0));
                        surface.shape_masks += 1;
                    }
                    DrawSegment::ApplyEffect(effect) => {
                        assert!(!has_target);
                        assert_eq!(
                            effect.parameters.bytes(&plan.effect_parameters),
                            &[1, 2, 3, 4]
                        );
                        surface.effects.push(effect.effect_id);
                    }
                    DrawSegment::Draws { instructions, .. } => {
                        assert!(has_target);
                        for instruction in &plan.instructions[instructions.clone()] {
                            let shape = match instruction.operation {
                                DrawOperation::DrawShape(draw)
                                | DrawOperation::DrawShapeAndIncrementStencil(draw)
                                | DrawOperation::DecrementStencil(draw) => draw.id,
                                DrawOperation::IncrementStencil(shape) => shape,
                                DrawOperation::CompositeTexture(_) => continue,
                            };
                            assert!(self.registered_shapes.contains(&shape.0));
                            surface.draws.push(shape.0);
                        }
                    }
                    DrawSegment::CaptureBackdrop(_) => panic!("unexpected backdrop"),
                }
            }
            assert!(!has_target, "backend receives complete target scopes");
        }
        Ok(())
    }
}

fn queue_shape(planner: &mut Planner, with_effects: bool) -> usize {
    let shape = CachedShapeHandle::new(
        &Shape::rect([(0.0, 0.0), (16.0, 16.0)], Stroke::default()),
        &mut planner.tessellator,
        &mut planner.shape_resources,
        Some(1),
    );
    let id = planner
        .draw_tree
        .add_node(DrawTreeNode::CachedShape(CachedShapeDrawData::new(
            shape,
            &ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        )));
    if with_effects {
        planner.shape_effects.insert(
            id,
            ShapeEffectInstance {
                effect_id: 7,
                params: Arc::from([1, 2, 3, 4]),
                config: ShapeEffectConfig::default(),
            },
        );
        planner.group_effects.insert(
            id,
            EffectInstance {
                effect_id: 8,
                params: vec![1, 2, 3, 4],
            },
        );
    }
    id
}

fn renderer() -> Renderer<'static, TestBackend> {
    Renderer {
        planner: Planner::new(
            Arc::new(RwLock::new(HashMap::new())),
            4096,
            DEFAULT_FRINGE_WIDTH,
        ),
        surface: TestSurface::default(),
        backend: TestBackend::default(),
        viewport: Viewport {
            physical_size: (32, 32),
            scale_factor: 1.0,
        },
        #[cfg(feature = "render_metrics")]
        last_planning_time: Duration::ZERO,
        #[cfg(feature = "render_metrics")]
        render_loop_metrics_tracker: Default::default(),
    }
}

#[test]
fn renderer_submits_completed_commands_to_its_own_surface_and_reuses_storage_after_rebuilds() {
    let mut renderer = renderer();
    let shape = queue_shape(&mut renderer.planner, true);
    renderer.backend.registered_shapes.push(shape);
    let planned_address = renderer.planner.plan(renderer.viewport) as *const RenderPlan as usize;
    renderer.render().unwrap();
    assert_eq!(renderer.backend.command_address, planned_address);
    assert_eq!(renderer.surface.draws, [shape]);
    assert_eq!(renderer.surface.shape_masks, 1);
    assert_eq!(renderer.surface.effects, [7, 8]);
    let instruction_address = renderer.backend.instruction_address;
    let mask_address = renderer.backend.mask_address;

    for _ in 0..3 {
        renderer.planner.clear_draw_queue();
        let rebuilt = queue_shape(&mut renderer.planner, true);
        assert_eq!(rebuilt, shape);
        renderer.render().unwrap();
        assert_eq!(renderer.backend.command_address, planned_address);
        assert_eq!(renderer.backend.instruction_address, instruction_address);
        assert_eq!(renderer.backend.mask_address, mask_address);
        assert_eq!(renderer.surface.effects, [7, 8]);
    }

    renderer.planner.clear_draw_queue();
    renderer.render().unwrap();
    assert!(renderer.surface.draws.is_empty());
    assert!(renderer.surface.effects.is_empty());
    assert_eq!(renderer.surface.shape_masks, 0);
    let plan = renderer.planner.plan(renderer.viewport);
    assert!(matches!(
        plan.scene.segments.as_slice(),
        [
            DrawSegment::BeginTarget(Target::Surface),
            DrawSegment::EndTarget
        ]
    ));
}

#[test]
fn one_completed_plan_targets_each_supplied_surface_and_backend_errors_reach_the_caller() {
    let mut first = renderer();
    let shape = queue_shape(&mut first.planner, false);
    first.backend.registered_shapes.push(shape);
    let mut second = renderer();
    second.backend.registered_shapes.push(shape);
    let commands = first.planner.plan(first.viewport);
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

    second.backend.should_fail = true;
    assert_eq!(second.render(), Err(TestRenderError));
    assert_eq!(second.surface.draws, [shape]);
    second.backend.should_fail = false;
    second.render().unwrap();
    assert!(second.surface.draws.is_empty());
    assert_eq!(first.surface.draws, [shape]);
}
