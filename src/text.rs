use crate::renderer::MathRect;
use crate::Color;
use glyphon::cosmic_text::Align;
use glyphon::{Attrs, Family, FontSystem, Metrics, Shaping, SwashCache, TextAtlas, TextRenderer};
use glyphon::{Buffer as TextBuffer, Color as TextColor, TextArea, TextBounds};
use wgpu::{Device, MultisampleState};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextAlignment {
    Start,
    End,
    Center,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextLayout {
    pub font_size: f32,
    pub line_height: f32,
    pub color: Color,
    pub area: MathRect,
    pub horizontal_alignment: TextAlignment,
    pub vertical_alignment: TextAlignment,
}

pub(crate) struct TextRendererWrapper {
    pub(crate) text_renderer: TextRenderer,
    pub(crate) atlas: TextAtlas,
    pub(crate) font_system: FontSystem,
    pub(crate) swash_cache: SwashCache,
}

impl TextRendererWrapper {
    pub fn new(
        device: &Device,
        queue: &wgpu::Queue,
        swapchain_format: wgpu::TextureFormat,
        depth_stencil_state: Option<wgpu::DepthStencilState>,
    ) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let mut atlas = TextAtlas::new(device, queue, swapchain_format);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            MultisampleState::default(),
            depth_stencil_state,
        );

        Self {
            text_renderer,
            atlas,
            font_system,
            swash_cache,
        }
    }
}

#[derive(Debug)]
pub(crate) struct TextDrawData {
    pub(crate) text_buffer: TextBuffer,
    pub(crate) area: MathRect,
    pub(crate) vertical_alignment: TextAlignment,
    pub(crate) data: String,
    pub(crate) font_size: f32,
    pub(crate) top: f32,
    pub(crate) clip_to_shape: Option<usize>,
}

impl TextDrawData {
    pub fn new(
        text: &str,
        layout: impl Into<TextLayout>,
        clip_to_shape: Option<usize>,
        scale_factor: f32,
        font_system: &mut FontSystem,
    ) -> Self {
        let layout = layout.into();

        let mut buffer = TextBuffer::new(
            font_system,
            Metrics::new(layout.font_size, layout.line_height),
        );

        let text_area_size = layout.area.size();

        buffer.set_size(font_system, text_area_size.width, text_area_size.height);

        // TODO: it's set text that causes performance issues
        buffer.set_text(
            font_system,
            text,
            Attrs::new()
                .family(Family::SansSerif)
                .metadata(clip_to_shape.unwrap()),
            Shaping::Advanced,
        );

        let align = match layout.horizontal_alignment {
            // None is equal to start of the line - left or right, depending on the language
            TextAlignment::Start => None,
            TextAlignment::End => Some(Align::End),
            TextAlignment::Center => Some(Align::Center),
        };

        for line in buffer.lines.iter_mut() {
            line.set_align(align);
        }

        let area = layout.area;

        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        buffer.shape_until_scroll(font_system);

        for layout_run in buffer.layout_runs() {
            for glyph in layout_run.glyphs.iter() {
                let physical_glyph = glyph.physical((0.0, 0.0), scale_factor);
                min_y = min_y.min(physical_glyph.y as f32 + layout_run.line_y);
                max_y = max_y.max(physical_glyph.y as f32 + layout_run.line_y);
            }
        }

        // TODO: that should be a line height
        let buffer_height = if max_y > min_y {
            max_y - min_y + layout.font_size
        } else {
            layout.font_size // for a single line
        };

        let top = match layout.vertical_alignment {
            TextAlignment::Start => area.min.y,
            TextAlignment::End => area.max.y - buffer_height,
            TextAlignment::Center => area.min.y + (area.height() - buffer_height) / 2.0,
        };

        // println!("Text {} is clipped to shape {}", text, clip_to_shape.unwrap_or(0));

        TextDrawData {
            top,
            text_buffer: buffer,
            area: layout.area,
            vertical_alignment: layout.vertical_alignment,
            data: text.to_string(),
            font_size: layout.font_size,
            clip_to_shape,
        }
    }
    pub fn to_text_area(&self, scale_factor: f32) -> TextArea {
        let area = self.area;
        let top = self.top;

        TextArea {
            buffer: &self.text_buffer,
            left: area.min.x * scale_factor,
            top: top * scale_factor,
            scale: scale_factor,
            bounds: TextBounds {
                left: (area.min.x * scale_factor) as i32,
                top: (area.min.y * scale_factor) as i32,
                right: (area.max.x * scale_factor) as i32,
                bottom: (area.max.y * scale_factor) as i32,
            },
            default_color: TextColor::rgb(255, 255, 255),
        }
    }
}
