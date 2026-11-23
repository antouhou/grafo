use super::TestTextureManager;
use crate::render_backend::texture_manager::{
    AlphaMode, TextureManager, TexturePixelLayout, TexturePixels, TextureStorageFormat,
    TextureStreamError,
};
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::{future, stream, StreamExt};
use std::ops::Range;
use std::rc::Rc;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("local texture source failed")]
struct LocalSourceFailure(Rc<()>);

struct LocalPixelChunk {
    pixels: Rc<[u8]>,
    range: Range<usize>,
}

impl AsRef<[u8]> for LocalPixelChunk {
    fn as_ref(&self) -> &[u8] {
        &self.pixels[self.range.clone()]
    }
}

fn upload_texture_streams<Manager: TextureManager>(manager: &Manager) {
    let pixels: Rc<[u8]> = Rc::from([12, 34, 56, 255, 78, 90, 123, 255]);
    let layout = TexturePixelLayout::rgba8_srgb((2, 1), AlphaMode::PremultipliedInLinearSpace);
    let (resume_sender, resume_receiver) = oneshot::channel();
    let final_pixels = Rc::clone(&pixels);
    let source = stream::iter([Ok::<_, LocalSourceFailure>(LocalPixelChunk {
        pixels: Rc::clone(&pixels),
        range: 0..3,
    })])
    .chain(stream::once(async move {
        resume_receiver.await.unwrap();
        Ok(LocalPixelChunk {
            pixels: final_pixels,
            range: 3..8,
        })
    }));
    let allocation =
        manager.allocate_texture_with_stream(42, TextureStorageFormat::Rgba8Srgb, layout, source);
    let (result, ()) = block_on(future::join(allocation, async move {
        resume_sender.send(()).unwrap();
    }));
    assert!(result.is_ok());
    assert!(manager.is_texture_loaded(42));

    let source = stream::iter([0..4, 4..8].into_iter().map(move |range| {
        Ok::<_, LocalSourceFailure>(LocalPixelChunk {
            pixels: Rc::clone(&pixels),
            range,
        })
    }));
    assert!(block_on(manager.stream_data_into_texture(42, (0, 0), layout, source)).is_ok());
}

#[test]
fn generic_texture_manager_accepts_local_stream_values() {
    upload_texture_streams(&TestTextureManager::default());
}

#[test]
fn local_source_failures_preserve_existing_texture_allocations() {
    let manager = TestTextureManager::default();
    manager.allocate_texture(42, (2, 1)).unwrap();
    let pixels: Rc<[u8]> = Rc::from([12, 34, 56, 255]);
    let failure = Rc::new(());

    for is_replacement in [true, false] {
        let source = stream::iter([
            Ok(LocalPixelChunk {
                pixels: Rc::clone(&pixels),
                range: 0..3,
            }),
            Err(LocalSourceFailure(Rc::clone(&failure))),
        ]);
        let size = if is_replacement { (4, 1) } else { (2, 1) };
        let layout = TexturePixelLayout::rgba8_srgb(size, AlphaMode::PremultipliedInLinearSpace);
        let result = if is_replacement {
            block_on(manager.allocate_texture_with_stream(
                42,
                TextureStorageFormat::Rgba8Srgb,
                layout,
                source,
            ))
        } else {
            block_on(manager.stream_data_into_texture(42, (0, 0), layout, source))
        };
        match result {
            Err(TextureStreamError::Source(LocalSourceFailure(returned_failure))) => {
                assert!(Rc::ptr_eq(&returned_failure, &failure));
            }
            _ => panic!("expected the original local source failure"),
        }
        let pixel =
            TexturePixels::rgba8_srgb(&pixels, (1, 1), AlphaMode::PremultipliedInLinearSpace);
        assert!(manager.load_data_into_texture(42, (1, 0), pixel).is_ok());
        assert!(manager.load_data_into_texture(42, (2, 0), pixel).is_err());
    }
}
