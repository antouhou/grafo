use super::composites::CompositeInstanceBuffer;
use super::draws::DrawPass;
use super::shapes::ShapeExecutionResources;
use super::targets;
use crate::renderer::commands::{DrawInstruction, DrawOperation, DrawPlan, TexturePlacement};

pub(super) fn execute_draw_instructions(
    commands: &DrawPlan,
    instructions: &[DrawInstruction],
    draw_pass: &mut DrawPass<'_, '_>,
    shapes: &ShapeExecutionResources,
    composite_instances: Option<CompositeInstanceBuffer>,
) {
    let mut index = 0;
    let mut composite_instance = 0;
    while let Some(instruction) = instructions.get(index) {
        if matches!(instruction.operation, DrawOperation::DrawShape(_)) {
            index += draw_pass.execute_leaf_draws(&instructions[index..], shapes);
            continue;
        }
        targets::set_scissor(draw_pass.render_pass, instruction.clip.scissor);
        match instruction.operation {
            DrawOperation::IncrementStencil(id) => draw_pass.increment_stencil(
                instruction.clip.stencil_reference,
                shapes.draw_resources(id),
            ),
            DrawOperation::DrawShapeAndIncrementStencil(draw) => draw_pass
                .draw_shape_and_increment_stencil(
                    instruction.clip.stencil_reference,
                    draw.material,
                    shapes.draw_resources(draw.id),
                ),
            DrawOperation::DecrementStencil(draw) => draw_pass.decrement_stencil(
                instruction.clip.stencil_reference,
                shapes.draw_resources(draw.id),
            ),
            DrawOperation::CompositeTexture(composite) => {
                match commands.composites[composite].placement {
                    TexturePlacement::Target => draw_pass.composite_texture(
                        instruction.clip.stencil_reference,
                        commands.composites[composite].texture,
                    ),
                    TexturePlacement::Local { .. } => {
                        let count = draw_pass.execute_texture_composites(
                            &instructions[index..],
                            &commands.composites,
                            &shapes.composites,
                            composite_instances.expect("local composites were prepared"),
                            composite_instance,
                        );
                        composite_instance += count as u32;
                        index += count;
                        continue;
                    }
                }
            }
            DrawOperation::DrawShape(_) => unreachable!("shape draws were consumed above"),
        }
        index += 1;
    }
}
