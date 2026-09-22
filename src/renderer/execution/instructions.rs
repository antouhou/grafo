use super::draws::DrawPass;
use super::shapes::ShapeExecutionResources;
use super::targets;
use crate::renderer::commands::{DrawInstruction, DrawOperation};

pub(super) fn execute_draw_instructions(
    instructions: &[DrawInstruction],
    draw_pass: &mut DrawPass<'_, '_>,
    shapes: &ShapeExecutionResources,
) {
    let mut index = 0;
    while let Some(instruction) = instructions.get(index) {
        if matches!(instruction.operation, DrawOperation::DrawShape(_)) {
            index += draw_pass.execute_leaf_draws(&instructions[index..], shapes);
            continue;
        }
        targets::set_scissor(draw_pass.render_pass, instruction.clip.scissor);
        match instruction.operation {
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
            DrawOperation::CompositeTexture(texture) => {
                draw_pass.composite_texture(instruction.clip.stencil_reference, texture);
            }
            DrawOperation::DrawShape(_) => unreachable!("shape draws were consumed above"),
        }
        index += 1;
    }
}
