//! Text rendering for the Grafo library.
//!
//! This module provides functionality to render text using the `glyphon` crate.
//!
//! # Examples
//!
//! Rendering text with specific layout:
//!
//! ```rust
//! use grafo::{TextAlignment, TextLayout};
//! use grafo::Color;
//! use grafo::MathRect;
//!
//! // Define the text layout
//! let layout = TextLayout {
//!     font_size: 16.0,
//!     line_height: 20.0,
//!     color: Color::rgb(255, 255, 255), // White text
//!     area: MathRect {
//!         min: (0.0, 0.0).into(),
//!         max: (200.0, 50.0).into(),
//!     },
//!     horizontal_alignment: TextAlignment::Center,
//!     vertical_alignment: TextAlignment::Center,
//! };
//!
//! // Usage of layout in rendering functions (pseudo-code)
//! // renderer.render_text("Hello, World!", layout);
//! ```

use crate::renderer::MathRect;
use crate::Color;
use glyphon::cosmic_text::Align;
use glyphon::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, TextAtlas, TextRenderer};
use glyphon::{Buffer as TextBuffer, Color as TextColor, TextArea, TextBounds};
use std::borrow::Cow;
use wgpu::{Device, MultisampleState};

/// Specifies the alignment of text within its layout area.
///
/// # Variants
///
/// - `Start`: Align text to the start (left or right, depending on language).
/// - `End`: Align text to the end.
/// - `Center`: Center-align the text.
///
/// # Examples
///
/// ```rust
/// use grafo::TextAlignment;
///
/// let align_start = TextAlignment::Start;
/// let align_end = TextAlignment::End;
/// let align_center = TextAlignment::Center;
/// ```
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextAlignment {
    /// Align text to the start (left or right, depending on language).
    Start,
    /// Align text to the end.
    End,
    /// Center-align the text.
    Center,
}

pub trait IntoCowBuffer<'a> {
    fn into_cow(self) -> std::borrow::Cow<'a, glyphon::Buffer>;
}

impl<'a> IntoCowBuffer<'a> for glyphon::Buffer {
    fn into_cow(self) -> std::borrow::Cow<'a, glyphon::Buffer> {
        std::borrow::Cow::Owned(self)
    }
}

impl<'a> IntoCowBuffer<'a> for &'a glyphon::Buffer {
    fn into_cow(self) -> std::borrow::Cow<'a, glyphon::Buffer> {
        std::borrow::Cow::Borrowed(self)
    }
}

/// Defines the layout parameters for rendering text.
///
/// # Fields
///
/// - `font_size`: The size of the font in pixels.
/// - `line_height`: The height of each line of text.
/// - `color`: The color of the text.
/// - `area`: The rectangular area within which the text is rendered.
/// - `horizontal_alignment`: The horizontal alignment of the text.
/// - `vertical_alignment`: The vertical alignment of the text.
///
/// # Examples
///
/// ```rust
/// use grafo::{TextAlignment, TextLayout};
/// use grafo::Color;
/// use grafo::MathRect;
///
/// let layout = TextLayout {
///     font_size: 16.0,
///     line_height: 20.0,
///     color: Color::rgb(255, 255, 255), // White text
///     area: MathRect {
///         min: (0.0, 0.0).into(),
///         max: (200.0, 50.0).into(),
///     },
///     horizontal_alignment: TextAlignment::Center,
///     vertical_alignment: TextAlignment::Center,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextLayout {
    /// The size of the font in pixels.
    pub font_size: f32,
    /// The height of each line of text.
    pub line_height: f32,
    /// The color of the text.
    pub color: Color,
    /// The rectangular area within which the text is rendered.
    pub area: MathRect,
    /// The horizontal alignment of the text.
    pub horizontal_alignment: TextAlignment,
    /// The vertical alignment of the text.
    pub vertical_alignment: TextAlignment,
}

/// Internal wrapper for `glyphon::TextRenderer` and related components.
///
/// This struct manages the text renderer, atlas, font system, and swash cache.
pub(crate) struct TextRendererWrapper {
    pub(crate) text_renderer: TextRenderer,
    pub(crate) atlas: TextAtlas,
    pub(crate) glyphon_cache: glyphon::Cache,
}

impl TextRendererWrapper {
    /// Creates a new `TextRendererWrapper`.
    ///
    /// # Parameters
    ///
    /// - `device`: The WGPU device.
    /// - `queue`: The WGPU queue.
    /// - `swapchain_format`: The format of the swapchain texture.
    /// - `depth_stencil_state`: Optional depth stencil state.
    pub fn new(
        device: &Device,
        queue: &wgpu::Queue,
        swapchain_format: wgpu::TextureFormat,
        depth_stencil_state: Option<wgpu::DepthStencilState>,
    ) -> Self {
        let glyphon_cache = glyphon::Cache::new(device);
        let mut atlas = TextAtlas::new(device, queue, &glyphon_cache, swapchain_format);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            device,
            MultisampleState::default(),
            depth_stencil_state,
        );

        Self {
            text_renderer,
            atlas,
            glyphon_cache,
        }
    }
}

pub struct NoopTextDataIter<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for NoopTextDataIter<'a> {
    type Item = TextDrawData<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[derive(Debug)]
pub struct TextDrawData<'a> {
    /// The text buffer containing glyph information.
    pub text_buffer: Cow<'a, TextBuffer>,
    /// The area within which the text is rendered.
    pub area: MathRect,
    /// The top position of the text within the layout area.
    pub top: f32,
    /// The color of the text.
    pub fallback_color: Color,
    /// Optional index of the shape to which the text is clipped.
    pub clip_to_shape: Option<usize>,
    pub buffer_metadata: usize,
}

impl<'a> TextDrawData<'a> {
    pub fn new(
        text: &str,
        layout: impl Into<TextLayout>,
        scale_factor: f32,
        font_system: &mut FontSystem,
        font_family: Family,
        clip_to_shape: Option<usize>,
        buffer_id: usize,
    ) -> Self {
        let layout = layout.into();

        let mut buffer = Buffer::new(
            font_system,
            Metrics::new(layout.font_size, layout.line_height),
        );

        let text_area_size = layout.area.size();

        buffer.set_size(
            font_system,
            Some(text_area_size.width),
            Some(text_area_size.height),
        );

        // TODO: it's set text that causes performance issues
        buffer.set_text(
            font_system,
            text,
            &Attrs::new()
                .family(font_family)
                .metadata(clip_to_shape.unwrap_or(0)),
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

        buffer.shape_until_scroll(font_system, false);

        for layout_run in buffer.layout_runs() {
            for glyph in layout_run.glyphs.iter() {
                let physical_glyph = glyph.physical((0.0, 0.0), scale_factor);
                min_y = min_y.min(physical_glyph.y as f32 + layout_run.line_y);
                max_y = max_y.max(physical_glyph.y as f32 + layout_run.line_y);
            }
        }

        // TODO: that should be a line height
        // let buffer_height = if max_y > min_y {
        //     max_y - min_y + layout.font_size
        // } else {
        //     layout.font_size // for a single line
        // };

        let buffer_height = buffer.size();
        let buffer_height = buffer_height.1.unwrap_or(0.0);

        println!("Buffer height: {buffer_height}");

        let top = match layout.vertical_alignment {
            TextAlignment::Start => area.min.y,
            TextAlignment::End => area.max.y - buffer_height,
            TextAlignment::Center => area.min.y + (area.height() - buffer_height) / 2.0,
        };

        // println!("Text {} is clipped to shape {}", text, clip_to_shape.unwrap_or(0));

        TextDrawData {
            top,
            text_buffer: Cow::Owned(buffer),
            area: layout.area,
            fallback_color: layout.color,
            clip_to_shape,
            buffer_metadata: buffer_id,
        }
    }

    pub fn with_buffer(
        buffer: impl IntoCowBuffer<'a>,
        area: MathRect,
        fallback_color: Color,
        vertical_offset: f32,
        clip_to_shape: Option<usize>,
        buffer_metadata: usize,
    ) -> Self {
        TextDrawData {
            top: vertical_offset,
            text_buffer: buffer.into_cow(),
            area,
            fallback_color,
            clip_to_shape,
            buffer_metadata,
        }
    }

    pub fn into_text_area(self, scale_factor: f32) -> TextArea<'a> {
        let area = self.area;
        let top = self.top;

        let bounds = TextBounds {
            left: (area.min.x * scale_factor) as i32,
            top: (area.min.y * scale_factor) as i32,
            right: (area.max.x * scale_factor) as i32,
            bottom: (area.max.y * scale_factor) as i32,
        };

        match self.text_buffer {
            Cow::Borrowed(buffer_ref) => TextArea {
                buffer: buffer_ref,
                left: area.min.x * scale_factor,
                top: top * scale_factor,
                scale: scale_factor,
                bounds,
                default_color: TextColor::rgba(
                    self.fallback_color.0[1],
                    self.fallback_color.0[2],
                    self.fallback_color.0[3],
                    self.fallback_color.0[0],
                ),
                custom_glyphs: &[],
            },
            Cow::Owned(_owned) => {
                panic!("Can't do right now");
            }
        }
    }

    pub fn to_text_area(&self, scale_factor: f32) -> TextArea {
        let area = self.area;
        let top = self.top;

        let bounds = TextBounds {
            left: (area.min.x * scale_factor) as i32,
            top: (area.min.y * scale_factor) as i32,
            right: (area.max.x * scale_factor) as i32,
            bottom: (area.max.y * scale_factor) as i32,
        };

        TextArea {
            buffer: &self.text_buffer,
            left: area.min.x * scale_factor,
            top: top * scale_factor,
            scale: scale_factor,
            bounds,
            default_color: TextColor::rgba(
                self.fallback_color.0[1],
                self.fallback_color.0[2],
                self.fallback_color.0[3],
                self.fallback_color.0[0],
            ),
            custom_glyphs: &[],
        }
    }
}
