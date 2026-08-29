use super::{validate_texture_region, TextureManagerError};

#[test]
fn texture_region_validation_accepts_regions_inside_the_texture() {
    assert!(validate_texture_region(7, (4, 3), (0, 0), (4, 3), 48).is_ok());
    assert!(validate_texture_region(7, (4, 3), (1, 1), (3, 2), 24).is_ok());
    assert!(matches!(
        validate_texture_region(7, (4, 3), (0, 0), (5, 3), 60),
        Err(TextureManagerError::InvalidTextureData(7))
    ));
    assert!(matches!(
        validate_texture_region(7, (4, 3), (0, 0), (4, 3), 47),
        Err(TextureManagerError::InvalidTextureData(7))
    ));
    assert!(matches!(
        validate_texture_region(7, (4, 3), (0, 0), (4, 3), 49),
        Err(TextureManagerError::InvalidTextureData(7))
    ));
}
