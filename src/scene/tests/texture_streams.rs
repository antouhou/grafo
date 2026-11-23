use super::TestTextureManager;
use crate::render_backend::texture_manager::{
    AlphaMode, TextureManager, TexturePixelLayout, TextureStorageFormat,
};
use futures::executor::block_on;
use futures::stream;
use std::rc::Rc;

fn upload_texture_streams<Manager: TextureManager>(manager: &Manager) {
    let pixels: Rc<[u8]> = Rc::from([12, 34, 56, 255, 78, 90, 123, 255]);
    let layout = TexturePixelLayout::rgba8_srgb((2, 1), AlphaMode::PremultipliedInLinearSpace);
    let source = stream::once(async { Ok::<_, Rc<()>>(Rc::clone(&pixels)) });
    let allocation =
        manager.allocate_texture_with_stream(42, TextureStorageFormat::Rgba8Srgb, layout, source);
    assert!(block_on(allocation).is_ok());
    assert!(manager.is_texture_loaded(42));

    let source = stream::once(async move { Ok::<_, Rc<()>>(pixels) });
    assert!(block_on(manager.stream_data_into_texture(42, (0, 0), layout, source)).is_ok());
}

#[test]
fn generic_texture_manager_accepts_local_stream_values() {
    upload_texture_streams(&TestTextureManager::default());
}
