// Removes texture-copy row padding on the GPU before readback.
// The CPU receives packed ARGB words and copies them without a per-row loop.

struct RowPackingParams {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    _pad: u32, // uniform layout padding
};

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_argb: array<u32>;

@group(0) @binding(2)
var<uniform> params: RowPackingParams;

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x >= params.width || invocation_id.y >= params.height) {
        return;
    }

    let input_row_offset = (invocation_id.y * params.padded_bytes_per_row) / 4u;
    let input_index = input_row_offset + invocation_id.x; // 1 word per pixel (4 bytes)
    let output_index = invocation_id.y * params.width + invocation_id.x;
    output_argb[output_index] = input_words[input_index];
}
