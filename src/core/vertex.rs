use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CustomVertex {
    pub(crate) position: [f32; 2],
    pub(crate) tex_coords: [f32; 2],
    /// Outward model-space normal used by the shader to offset the AA fringe.
    pub(crate) normal: [f32; 2],
    /// Coverage is 1.0 at interior and boundary vertices, and 0.0 at the outer fringe.
    pub(crate) coverage: f32,
}

impl CustomVertex {
    pub(crate) fn unit_quad() -> [Self; 4] {
        let [(minimum_x, minimum_y), (maximum_x, maximum_y)] = [(0.0, 0.0), (1.0, 1.0)];
        [
            Self {
                position: [minimum_x, minimum_y],
                tex_coords: [0.0, 0.0],
                normal: [0.0; 2],
                coverage: 1.0,
            },
            Self {
                position: [maximum_x, minimum_y],
                tex_coords: [1.0, 0.0],
                normal: [0.0; 2],
                coverage: 1.0,
            },
            Self {
                position: [maximum_x, maximum_y],
                tex_coords: [1.0, 1.0],
                normal: [0.0; 2],
                coverage: 1.0,
            },
            Self {
                position: [minimum_x, maximum_y],
                tex_coords: [0.0, 1.0],
                normal: [0.0; 2],
                coverage: 1.0,
            },
        ]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct InstanceTransform {
    pub col0: [f32; 4],
    pub col1: [f32; 4],
    pub col2: [f32; 4],
    pub col3: [f32; 4],
}

impl InstanceTransform {
    pub fn identity() -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a 2D translation transform (tx, ty) in pixels.
    ///
    /// Fields `col0` through `col3` store the columns of the GPU `mat4x4`.
    /// Translation lives in `col3`.
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [tx, ty, 0.0, 1.0],
        }
    }

    /// Create a 3D translation transform (tx, ty, tz).
    ///
    /// Translation is stored in `col3`.
    pub fn translation3d(tx: f32, ty: f32, tz: f32) -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [tx, ty, tz, 1.0],
        }
    }

    /// Create a 2D scale transform with factors (sx, sy).
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            col0: [sx, 0.0, 0.0, 0.0],
            col1: [0.0, sy, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a 3D scale transform with factors (sx, sy, sz).
    pub fn scale3d(sx: f32, sy: f32, sz: f32) -> Self {
        Self {
            col0: [sx, 0.0, 0.0, 0.0],
            col1: [0.0, sy, 0.0, 0.0],
            col2: [0.0, 0.0, sz, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation around the Z axis by `radians`.
    /// Positive angles rotate counter-clockwise in screen space.
    pub fn rotation_z(radians: f32) -> Self {
        let (s, c) = radians.sin_cos();
        // Column-major layout
        Self {
            col0: [c, -s, 0.0, 0.0],
            col1: [s, c, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation around the Z axis by `degrees`.
    pub fn rotation_z_deg(degrees: f32) -> Self {
        Self::rotation_z(degrees.to_radians())
    }

    /// Create a 2D affine transform from matrix components:
    ///   [ a c tx ]
    ///   [ b d ty ]
    ///   [ 0 0  1 ]
    ///
    /// Stored so that `col0..col3` map to GPU columns 0..3:
    ///   `col0=[a,b,0,0]  col1=[c,d,0,0]  col2=[0,0,1,0]  col3=[tx,ty,0,1]`
    pub fn affine_2d(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> Self {
        Self {
            col0: [a, b, 0.0, 0.0],
            col1: [c, d, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [tx, ty, 0.0, 1.0],
        }
    }

    /// Composes transforms by applying `self` first, then `rhs`.
    ///
    /// The resulting matrix is `rhs * self`.
    pub fn multiply(&self, rhs: &Self) -> Self {
        // Dot product of a column from self with a row from rhs.
        fn dot(col: [f32; 4], mat: &InstanceTransform, row_index: usize) -> f32 {
            let other = [
                mat.col0[row_index],
                mat.col1[row_index],
                mat.col2[row_index],
                mat.col3[row_index],
            ];
            col[0] * other[0] + col[1] * other[1] + col[2] * other[2] + col[3] * other[3]
        }

        Self {
            col0: [
                dot(self.col0, rhs, 0),
                dot(self.col0, rhs, 1),
                dot(self.col0, rhs, 2),
                dot(self.col0, rhs, 3),
            ],
            col1: [
                dot(self.col1, rhs, 0),
                dot(self.col1, rhs, 1),
                dot(self.col1, rhs, 2),
                dot(self.col1, rhs, 3),
            ],
            col2: [
                dot(self.col2, rhs, 0),
                dot(self.col2, rhs, 1),
                dot(self.col2, rhs, 2),
                dot(self.col2, rhs, 3),
            ],
            col3: [
                dot(self.col3, rhs, 0),
                dot(self.col3, rhs, 1),
                dot(self.col3, rhs, 2),
                dot(self.col3, rhs, 3),
            ],
        }
    }

    /// Compose two transforms so that `self` is applied first, then `next`.
    ///
    /// Alias for [`multiply`](Self::multiply) that mirrors euclid's `Transform3D::then()`.
    ///
    /// ```rust
    /// use grafo::TransformInstance;
    ///
    /// let rotation = TransformInstance::rotation_z_deg(45.0);
    /// let translation = TransformInstance::translation(10.0, 20.0);
    /// let composed = rotation.then(&translation); // rotate first, translate second
    /// ```
    pub fn then(&self, next: &Self) -> Self {
        self.multiply(next)
    }

    /// Return the 4 columns as an array.
    pub fn as_cols(&self) -> [[f32; 4]; 4] {
        [self.col0, self.col1, self.col2, self.col3]
    }

    /// Build from 4 columns.
    pub fn from_cols(cols: [[f32; 4]; 4]) -> Self {
        Self {
            col0: cols[0],
            col1: cols[1],
            col2: cols[2],
            col3: cols[3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TextureUvTransform {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

impl TextureUvTransform {
    pub const IDENTITY: Self = Self {
        scale: [1.0, 1.0],
        offset: [0.0, 0.0],
    };
}

impl Default for TextureUvTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}
