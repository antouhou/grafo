use super::{PixelFormat, PixelLayout, Pixmap, PixmapMut, RenderTargetError};

#[test]
fn borrowed_storage_validates_stride_size_and_overflow() {
    assert_eq!(
        PixelLayout::tightly_packed((0, 4), PixelFormat::Bgra8),
        Err(RenderTargetError::InvalidSize)
    );
    assert!(matches!(
        PixelLayout::new((3, 2), PixelFormat::Bgra8, 11),
        Err(RenderTargetError::InvalidStride { .. })
    ));
    assert!(matches!(
        PixelLayout::new((1, 3), PixelFormat::Bgra8, usize::MAX),
        Err(RenderTargetError::InvalidSize)
    ));
    let layout = PixelLayout::new((3, 2), PixelFormat::Bgra8, 16).unwrap();
    let mut storage = [71; 27];
    assert!(matches!(
        PixmapMut::new(&mut storage, layout),
        Err(RenderTargetError::StorageTooSmall {
            required: 28,
            provided: 27
        })
    ));
    assert_eq!(storage, [71; 27]);
    let mut storage = [71; 28];
    assert!(PixmapMut::new(&mut storage, layout).is_ok());
}

#[test]
fn owned_surface_reuses_storage_and_borrows_without_copying() {
    let mut pixels = Vec::with_capacity(1024);
    pixels.resize(28, 31);
    let address = pixels.as_ptr();
    let layout = PixelLayout::new((3, 2), PixelFormat::Bgra8, 16).unwrap();
    let mut surface = Pixmap::from_vec(pixels, layout).unwrap();
    surface.resize((8, 8)).unwrap();
    assert_eq!(surface.pixels().as_ptr(), address);
    let mut borrowed = surface.as_mut();
    borrowed.pixels_mut()[0] = 99;
    assert_eq!(surface.pixels()[0], 99);
    let previous_layout = surface.layout();
    assert_eq!(surface.resize((0, 8)), Err(RenderTargetError::InvalidSize));
    assert_eq!(surface.layout(), previous_layout);
    assert_eq!(surface.pixels()[0], 99);
}
