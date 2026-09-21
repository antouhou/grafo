use crate::renderer::passes::{self, PendingLeafBatch};
use crate::renderer::types::DrawTreeNode;
use crate::shape::CachedShapeDrawData;
use crate::{
    Color, Fill, Gradient, GradientStop, GradientStopOffset, LinearGradientDesc,
    LinearGradientLine, Renderer, Shape, ShapeDrawCommandOptions, ShapeTextureFitMode, Stroke,
    TransformInstance,
};
use futures::executor;
use std::sync::Arc;

fn create_renderer() -> Renderer<'static> {
    executor::block_on(Renderer::try_new_headless((64, 64), 1.0))
        .expect("shape execution tests require a GPU adapter")
}

fn shape_description<'a>(renderer: &'a Renderer<'_>, node_id: usize) -> &'a CachedShapeDrawData {
    match renderer.state.draw_tree.get(node_id).unwrap() {
        DrawTreeNode::CachedShape(shape) => shape,
        DrawTreeNode::ClipRect(_) => panic!("expected a shape description"),
    }
}

fn queue_material_shapes(renderer: &mut Renderer<'_>) -> [usize; 7] {
    let clip_id = renderer
        .add_clipping_rect(
            [(0.0, 0.0), (64.0, 64.0)],
            None,
            None::<TransformInstance>,
            true,
        )
        .unwrap();
    renderer
        .add_shape(
            Shape::builder().build(),
            Some(clip_id),
            None,
            ShapeDrawCommandOptions::new(),
        )
        .unwrap();
    let shape_key = 400;
    renderer.load_shape(
        Shape::rect([(0.0, 0.0), (18.0, 18.0)], Stroke::default()),
        shape_key,
        Some(shape_key),
    );
    let gradient = Gradient::linear(LinearGradientDesc::new(
        LinearGradientLine {
            start: [0.0, 0.0],
            end: [18.0, 0.0],
        },
        [
            GradientStop::at_position(GradientStopOffset::linear_radial(0.0), Color::BLACK),
            GradientStop::at_position(GradientStopOffset::linear_radial(1.0), Color::WHITE),
        ],
    ))
    .unwrap();
    let options = [
        ShapeDrawCommandOptions::new().color(Color::rgb(255, 0, 0)),
        ShapeDrawCommandOptions::new()
            .color(Color::rgb(0, 0, 255))
            .transform(TransformInstance::translation(20.0, 0.0)),
        ShapeDrawCommandOptions::new().background_texture_id(401),
        ShapeDrawCommandOptions::new()
            .background_texture_id(401)
            .background_texture_fit_mode(ShapeTextureFitMode::Contain),
        ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient.clone())),
        ShapeDrawCommandOptions::new().fill(Fill::Gradient(gradient)),
        ShapeDrawCommandOptions::new().color(Color::rgb(0, 255, 0)),
    ];
    options.map(|options| {
        renderer
            .add_cached_shape(shape_key, Some(clip_id), options)
            .unwrap()
    })
}

#[test]
fn shared_geometry_batches_keep_each_shapes_instance_data() {
    let mut renderer = create_renderer();
    renderer.texture_manager().allocate_texture_with_data(
        401,
        (2, 1),
        &[0, 255, 0, 255, 0, 255, 0, 255],
    );
    let node_ids = queue_material_shapes(&mut renderer);
    let execution = &renderer.state.shape_execution;
    let geometry_range = execution.draws[&node_ids[0]].geometry_buffer_range.unwrap();
    for (instance_index, node_id) in node_ids.iter().enumerate() {
        let resources = &execution.draws[node_id];
        assert_eq!(resources.geometry_buffer_range, Some(geometry_range));
        assert_eq!(resources.instance_index, Some(instance_index));
    }
    assert_eq!(execution.geometry_ranges.len(), 1);
    assert!(
        !execution.draws.contains_key(&0),
        "scissor nodes need no GPU draw data"
    );
    assert!(execution.draws[&1].geometry_buffer_range.is_none());
    assert!(execution.draws[&1].instance_index.is_none());
    assert_eq!(execution.instance_colors[0].color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(execution.instance_colors[1].color, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(execution.instance_transforms[1].col3, [20.0, 0.0, 0.0, 1.0]);
    assert_eq!(execution.instance_metadata[2].texture_flags, 1.0);
    assert_eq!(
        execution.instance_metadata[2]
            .texture_uv_transform_layer0
            .scale,
        [1.0, 1.0]
    );
    assert_eq!(
        execution.instance_metadata[3]
            .texture_uv_transform_layer0
            .scale,
        [1.0, 2.0]
    );

    let mut batch = PendingLeafBatch::default();
    for (node_id, should_batch) in node_ids[..3].iter().zip([true, true, false]) {
        assert_eq!(
            passes::try_batch_leaf(
                &mut batch,
                shape_description(&renderer, *node_id),
                &execution.draws[node_id],
                0
            ),
            should_batch,
        );
    }
    let mut batch = PendingLeafBatch::default();
    for (node_id, should_batch) in node_ids[2..5].iter().zip([true, true, false]) {
        assert_eq!(
            passes::try_batch_leaf(
                &mut batch,
                shape_description(&renderer, *node_id),
                &execution.draws[node_id],
                0
            ),
            should_batch,
        );
    }
    assert!(!passes::try_batch_leaf(
        &mut PendingLeafBatch::default(),
        shape_description(&renderer, node_ids[4]),
        &execution.draws[&node_ids[4]],
        0,
    ));
}

#[test]
fn queue_rebuild_reuses_storage_and_gradients_until_pipeline_recreation() {
    let mut renderer = create_renderer();
    let node_ids = queue_material_shapes(&mut renderer);
    let execution = &renderer.state.shape_execution;
    let gradient_binding = Arc::clone(
        execution.draws[&node_ids[4]]
            .gradient_bind_group
            .as_ref()
            .unwrap(),
    );
    assert!(Arc::ptr_eq(
        &gradient_binding,
        execution.draws[&node_ids[5]]
            .gradient_bind_group
            .as_ref()
            .unwrap(),
    ));
    let vertices_pointer = execution.vertices.as_ptr();
    let instances_pointer = execution.instance_transforms.as_ptr();
    let draw_capacity = execution.draws.capacity();
    let mut pixels = Vec::new();
    renderer.render_to_buffer(&mut pixels).unwrap();
    let first_pixels = pixels.clone();
    renderer.clear_draw_queue();
    let node_ids = queue_material_shapes(&mut renderer);
    let execution = &renderer.state.shape_execution;
    assert_eq!(execution.vertices.as_ptr(), vertices_pointer);
    assert_eq!(execution.instance_transforms.as_ptr(), instances_pointer);
    assert_eq!(execution.draws.capacity(), draw_capacity);
    assert_eq!(execution.draws[&node_ids[0]].instance_index, Some(0));
    assert!(Arc::ptr_eq(
        &gradient_binding,
        execution.draws[&node_ids[4]]
            .gradient_bind_group
            .as_ref()
            .unwrap(),
    ));
    renderer.render_to_buffer(&mut pixels).unwrap();
    assert_eq!(pixels, first_pixels);

    renderer.set_msaa_samples(4);
    assert!(!Arc::ptr_eq(
        &gradient_binding,
        renderer.state.shape_execution.draws[&node_ids[4]]
            .gradient_bind_group
            .as_ref()
            .unwrap(),
    ));
    renderer.render_to_buffer(&mut pixels).unwrap();
}
