// The input may contain unused bytes at the end of each image row.
// Skip those bytes when copying the pixels to the output. Pixel values stay unchanged.

struct Params {
    width: u32,
    height: u32,
    padded_bpr: u32, // bytes per row including padding; guaranteed multiple of 256 and 4
    _pad: u32,       // std140 alignment padding
};

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_argb: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let row_words = (gid.y * params.padded_bpr) / 4u;
    let word_index = row_words + gid.x; // 1 word per pixel (4 bytes)
    let out_index = gid.y * params.width + gid.x;
    output_argb[out_index] = input_words[word_index];
}
