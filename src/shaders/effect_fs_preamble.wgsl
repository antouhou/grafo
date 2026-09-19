// Input texture bindings prepended to the user's effect fragment shader.

// The engine binds the input texture and sampler at group 0.
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
