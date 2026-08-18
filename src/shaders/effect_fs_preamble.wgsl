// Built-in fragment shader preamble providing the input texture bindings.
// This is prepended to the user's effect fragment shader.

// -- Provided by the engine (group 0) --
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
