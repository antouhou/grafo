use super::Renderer;
use crate::core::vertex::{CustomVertex, InstanceTransform};
use crate::renderer::backend::vertex::{InstanceColor, InstanceMetadata};
use std::mem;

const INDEX_ELEMENT_SIZE: usize = mem::size_of::<u16>();

impl Renderer<'_> {
    pub fn print_memory_usage_info(&self) {
        println!("=== Memory Usage Info ===");

        println!(
            "Cached shapes: {}",
            self.planner
                .loaded_shapes
                .read()
                .expect("shared shape cache lock poisoned")
                .len()
        );
        println!("Draw tree size: {}", self.planner.draw_tree.len());

        println!("\n--- Temporary Vectors ---");
        println!(
            "Temp vertices: {} items, {} capacity, ~{} bytes",
            self.backend.resources.shape_execution.vertices.len(),
            self.backend.resources.shape_execution.vertices.capacity(),
            self.backend.resources.shape_execution.vertices.capacity()
                * CustomVertex::STRIDE as usize
        );
        println!(
            "Temp indices: {} items, {} capacity, ~{} bytes",
            self.backend.resources.shape_execution.indices.len(),
            self.backend.resources.shape_execution.indices.capacity(),
            self.backend.resources.shape_execution.indices.capacity() * INDEX_ELEMENT_SIZE
        );
        println!(
            "Temp instance transforms: {} items, {} capacity, ~{} bytes",
            self.backend
                .resources
                .shape_execution
                .instance_transforms
                .len(),
            self.backend
                .resources
                .shape_execution
                .instance_transforms
                .capacity(),
            self.backend
                .resources
                .shape_execution
                .instance_transforms
                .capacity()
                * InstanceTransform::STRIDE as usize
        );
        println!(
            "Temp instance colors: {} items, {} capacity, ~{} bytes",
            self.backend.resources.shape_execution.instance_colors.len(),
            self.backend
                .resources
                .shape_execution
                .instance_colors
                .capacity(),
            self.backend
                .resources
                .shape_execution
                .instance_colors
                .capacity()
                * InstanceColor::STRIDE as usize
        );
        println!(
            "Temp instance metadata: {} items, {} capacity, ~{} bytes",
            self.backend
                .resources
                .shape_execution
                .instance_metadata
                .len(),
            self.backend
                .resources
                .shape_execution
                .instance_metadata
                .capacity(),
            self.backend
                .resources
                .shape_execution
                .instance_metadata
                .capacity()
                * InstanceMetadata::STRIDE as usize
        );

        println!("\n--- GPU Buffers ---");
        let buffers = &self.backend.resources.buffers;
        if let Some(buf) = &buffers.aggregated_vertex_buffer {
            println!("Aggregated vertex buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_index_buffer {
            println!("Aggregated index buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_instance_transform_buffer {
            println!("Aggregated instance transform buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_instance_color_buffer {
            println!("Aggregated instance color buffer: {} bytes", buf.size());
        }
        if let Some(buf) = &buffers.aggregated_instance_metadata_buffer {
            println!("Aggregated instance metadata buffer: {} bytes", buf.size());
        }

        println!("\n--- ARGB Compute Buffers ---");
        if let Some(resources) = &self.backend.argb_readback {
            let target = &resources.target;
            println!("ARGB input buffer: {} bytes", target.input_buffer.size());
            println!(
                "ARGB output storage buffer: {} bytes",
                target.output_buffer.size()
            );
            println!(
                "ARGB readback buffer: {} bytes",
                target.readback_buffer.size()
            );
            println!("ARGB params buffer: {} bytes", target.params_buffer.size());
            println!(
                "ARGB offscreen texture: {}x{}",
                target.texture.width(),
                target.texture.height()
            );
        }

        println!("\n--- Render-to-Buffer Caches ---");
        if let Some(resources) = &self.backend.bgra_readback {
            println!(
                "RTB offscreen texture: {}x{}",
                resources.texture.width(),
                resources.texture.height()
            );
            println!("RTB readback buffer: {} bytes", resources.buffer.size());
        }

        println!("\n--- Uniform Buffers ---");
        println!(
            "AND uniform buffer: {} bytes",
            self.backend
                .pipeline_resources
                .shapes
                .and_uniform_buffer
                .size()
        );
        println!(
            "Decrementing uniform buffer: {} bytes",
            self.backend
                .pipeline_resources
                .shapes
                .decrementing_uniform_buffer
                .size()
        );

        println!("\n--- Texture Manager ---");
        println!(
            "{:?}",
            self.backend
                .pipeline_resources
                .shapes
                .texture_manager
                .size()
        );

        println!("\n--- Shape Resources ---");
        self.planner.shape_resources.print_sizes();
        self.backend
            .resources
            .shape_execution
            .gradient_cache
            .print_sizes();

        println!("=========================");
    }
}
