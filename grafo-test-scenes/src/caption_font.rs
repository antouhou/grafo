//! A small 5-by-7 pixel font for example captions and diagnostics.

use grafo::Shape;

fn glyph_rows(character: u8) -> [u8; 7] {
    match character {
        b'A' => [14, 17, 17, 31, 17, 17, 17],
        b'B' => [30, 17, 17, 30, 17, 17, 30],
        b'C' => [14, 17, 16, 16, 16, 17, 14],
        b'D' => [30, 17, 17, 17, 17, 17, 30],
        b'E' => [31, 16, 16, 30, 16, 16, 31],
        b'F' => [31, 16, 16, 30, 16, 16, 16],
        b'G' => [14, 17, 16, 23, 17, 17, 15],
        b'H' => [17, 17, 17, 31, 17, 17, 17],
        b'I' => [31, 4, 4, 4, 4, 4, 31],
        b'J' => [7, 2, 2, 2, 18, 18, 12],
        b'K' => [17, 18, 20, 24, 20, 18, 17],
        b'L' => [16, 16, 16, 16, 16, 16, 31],
        b'M' => [17, 27, 21, 21, 17, 17, 17],
        b'N' => [17, 25, 25, 21, 19, 19, 17],
        b'O' => [14, 17, 17, 17, 17, 17, 14],
        b'P' => [30, 17, 17, 30, 16, 16, 16],
        b'Q' => [14, 17, 17, 17, 21, 18, 13],
        b'R' => [30, 17, 17, 30, 20, 18, 17],
        b'S' => [15, 16, 16, 14, 1, 1, 30],
        b'T' => [31, 4, 4, 4, 4, 4, 4],
        b'U' => [17, 17, 17, 17, 17, 17, 14],
        b'V' => [17, 17, 17, 17, 17, 10, 4],
        b'W' => [17, 17, 17, 21, 21, 21, 10],
        b'X' => [17, 17, 10, 4, 10, 17, 17],
        b'Y' => [17, 17, 10, 4, 4, 4, 4],
        b'Z' => [31, 1, 2, 4, 8, 16, 31],
        b'0' => [14, 17, 19, 21, 25, 17, 14],
        b'1' => [4, 12, 4, 4, 4, 4, 14],
        b'2' => [14, 17, 1, 2, 4, 8, 31],
        b'3' => [30, 1, 1, 14, 1, 1, 30],
        b'4' => [2, 6, 10, 18, 31, 2, 2],
        b'5' => [31, 16, 16, 30, 1, 1, 30],
        b'6' => [14, 16, 16, 30, 17, 17, 14],
        b'7' => [31, 1, 2, 4, 8, 8, 8],
        b'8' => [14, 17, 17, 14, 17, 17, 14],
        b'9' => [14, 17, 17, 15, 1, 1, 14],
        b'+' => [0, 4, 4, 31, 4, 4, 0],
        b'-' => [0, 0, 0, 31, 0, 0, 0],
        b'.' => [0, 0, 0, 0, 0, 0, 4],
        b':' => [0, 4, 0, 0, 4, 0, 0],
        b'%' => [25, 26, 2, 4, 8, 11, 19],
        b'/' => [1, 2, 2, 4, 8, 8, 16],
        _ => [0; 7],
    }
}

/// Width of a single line in font pixels, with one pixel between characters.
pub fn width(text: &str) -> f32 {
    text.len().saturating_mul(6).saturating_sub(1) as f32
}

/// Builds a single line at the origin. Supports uppercase ASCII, digits, and `+-.:%/`.
/// Unsupported bytes occupy a blank character cell.
pub fn shape(text: &str) -> Shape {
    let mut path = Shape::builder();
    for (column, character) in text.bytes().enumerate() {
        for (row, bits) in glyph_rows(character).into_iter().enumerate() {
            for pixel in 0..5 {
                if bits & (1 << (4 - pixel)) == 0 {
                    continue;
                }
                let x = (column * 6 + pixel) as f32;
                let y = row as f32;
                path = path
                    .begin((x, y))
                    .line_to((x + 1.0, y))
                    .line_to((x + 1.0, y + 1.0))
                    .line_to((x, y + 1.0))
                    .close();
            }
        }
    }
    path.build()
}
